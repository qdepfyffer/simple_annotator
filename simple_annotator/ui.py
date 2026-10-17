"""
Qt user interface for the annotator

MainWindow: Application window - File browser on the left, image canvas to the right, toolbar up top
            Has ownership over the AnnotationSession and user settings

ImageDirProxy:  Allows for filtering of the Qt filesystem model so we can hide directories that don't contain images
                and recolor entries in the file browser based on their status (active, annotated)

SegmentWorker:  What actually runs the slow segmentation - happens on a background thread so the window remains responsive
                and reports back via Qt signals

SettingsDialog: Allows for the user to choose a segmenter and edit its parameters + border color

The canvas is just a matplotlib figure. We let matplotlib handle pretty much everything to do with the display of the image.
It's really slow but since this was based on the previous tool it was a natural place to start. Future work could potentially
find another way to handle image display that isn't so horribly slow.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from PIL import Image
from PySide6.QtCore import (
    QDir,
    QModelIndex,
    QObject,
    QRunnable,
    QSortFilterProxyModel,
    Qt,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QCloseEvent,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPalette,
)
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileSystemModel,
    QFormLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from . import config, segmentation
from .annotation import DEFAULT_CLASSES, AnnotationSession
from .mask import count_annotated, mask_path

# What file extensions will be treated as valid images by our annotator?
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ANNOTATED_COLOR = QColor("#0072B2")  # Color of images w/ annotations already (provided they're where they're expected)
CURRENT_COLOR = QColor("#EB4034")   # Color of image that is in the process of being annotated, unused for now, ugly

# Define a sorting key so next/prev navigation follows the layout the user would expect based on the file tree
def _natural_key(path: Path) -> list[int | str]:
    """
    Sort key that matches the file tree - case insensitive & strings of digits compared as numbers
    """

    # Ex. img10.png becomes ["img", 10, ".png"]
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


class MainWindow(QMainWindow):
    """
    Main window for the annotator.
    """

    def __init__(self) -> None:
        """
        Build the whole window. 
        """

        # Basic window initialization, set the title & starting resolution, load settings
        super().__init__()
        self.setWindowTitle("Simple Annotator")
        self.resize(1280, 720)
        self.settings = config.load()

        self.pool = QThreadPool.globalInstance()            # Qt global thread pool
        self.session: AnnotationSession | None = None       # Image currently being annotated
        self._pending_path: Path | None = None              # Path to the image currently being segmented (so we can throw away results from prev images if we open a new image before it finishes)
        self._image_artist: AxesImage | None = None                           
        self._display_image: np.ndarray | None = None       # Full-res image currently on screen
        self._pan_anchor: tuple[float, float] | None = None # Image coordinate which pan movements are anchored to
        self._last_pan_draw = 0.0                           # For throttling redraws
        
        # === FILE BROWSER =============================================================================================
        self.fs_model = QFileSystemModel()                      # Directories lazily read and exposed as a tree model
        self.fs_model.setRootPath(QDir.homePath())              # Start from the user's home folder
        self.tree = QTreeView()                                 # Visible file tree widget
        self.proxy = ImageDirProxy(self.fs_model)               # Proxy for the filesystem that lets us hide irrelevant files / folders and lets us recolor entries
        self.tree.setModel(self.proxy)
        self.tree.setRootIndex(
            self.proxy.mapFromSource(self.fs_model.index(
                QDir.homePath())                                # Set the home folder as the top of the tree (we have to map it to the proxy for the view)
                )
            )
        for column in range(1, self.fs_model.columnCount()):
            self.tree.hideColumn(column)                        # Hide columns [1..] of the QFileSystemModel, this leaves only the name (column 0)
        self.tree.activated.connect(self._on_file_activated)    # "activated" fires on double-click (or enter) on most platforms 

        # === IMAGE CANVAS =============================================================================================
        background = self.palette().color(QPalette.ColorRole.Window).name()     # Make the figure background the same color as the Qt window
        self.figure = Figure(facecolor=background)                              # Actual drawing surface for matplotlib
        self.canvas = FigureCanvasQTAgg(self.figure)                            # Qt widget that displays said drawing surface

        # Connect the matplotlib mouse events to our handlers
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)    
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # Set the axes for our figure and turn their visuals off so they don't show on the canvas
        self.axes = self.figure.add_axes((0, 0, 1, 1))
        self.axes.set_axis_off()

        # Set up the layout - side by side file browser and canvas w/ a draggable divider
        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([300, 900])
        self.setCentralWidget(splitter)

        # First call of statusBar(), creates the bar and sets the initial message
        self.statusBar().showMessage("Open an image to begin")

        # === TOOLBAR ==================================================================================================
        toolbar = self.addToolBar("Annotate")
        toolbar.setMovable(False)  # User shouldn't be able to move the toolbar

        save_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.DocumentSave), "Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save)
        toolbar.addAction(save_action)

        prev_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.GoPrevious), "Previous Image", self)
        prev_action.triggered.connect(lambda: self._step_image(-1))
        prev_action.setShortcut("A")
        toolbar.addAction(prev_action)

        next_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.GoNext), "Next Image", self)
        next_action.triggered.connect(lambda: self._step_image(1))
        next_action.setShortcut("D")
        toolbar.addAction(next_action)

        toolbar.addSeparator()
        undo_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.EditUndo), "Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._undo)
        toolbar.addAction(undo_action)

        redo_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.EditRedo), "Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self._redo)
        toolbar.addAction(redo_action)

        clear_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.EditClear), "Clear Mask", self)
        clear_action.triggered.connect(self._clear_mask)
        toolbar.addAction(clear_action)

        toolbar.addSeparator()
        self.borders_action = QAction("Borders", self)
        self.borders_action.setCheckable(True)
        self.borders_action.setChecked(True)
        self.borders_action.setShortcut("B")
        self.borders_action.toggled.connect(self._toggle_borders)
        toolbar.addAction(self.borders_action)

        # One checkable option per class - mutually exclusive
        # Doing it like this makes it more extensible should we ever need more than 2 classes
        self.class_group = QActionGroup(self)
        for index, cls in enumerate(DEFAULT_CLASSES):
            action = QAction(cls.name, self)
            action.setCheckable(True)
            action.setShortcut(str(index + 1))
            action.setData(index)
            self.class_group.addAction(action)
            toolbar.addAction(action)
        self.class_group.actions()[1].setChecked(True)  # Sets the first non-default class as the class to paint

        toolbar.addSeparator()
        progress_action = QAction("Progress", self)
        progress_action.triggered.connect(self._show_progress)
        toolbar.addAction(progress_action)

        toolbar.addSeparator()
        reset_action = QAction("Reset View", self)
        reset_action.triggered.connect(self._reset_view)
        toolbar.addAction(reset_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        toolbar.addSeparator()
        settings_icon = QIcon.fromTheme("preferences-system", QIcon.fromTheme(QIcon.ThemeIcon.DocumentProperties))
        settings_action = QAction(settings_icon, "Settings", self)
        settings_action.triggered.connect(self._open_settings)
        toolbar.addAction(settings_action)




    def _on_file_activated(self, index: QModelIndex) -> None:
        path = Path(self.fs_model.filePath(self.proxy.mapToSource(index)))
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return
        self.open_image(path)


    def open_image(self, path: Path) -> None:
        self._finish_session()  # Save the previous image before opening the new one
        image = np.asarray(Image.open(path).convert("RGBA").convert("RGB"))
        self._image_artist = None  # Reset signal
        self._show(image)  # Shows raw img while segmentation is running
        self.statusBar().showMessage(f"Segmenting {path.name}...")

        self.session = None
        self.borders_action.setChecked(True)  # Reset border display option on image change so every segmentation immediately shows borders
        self._pending_path = path
        self.proxy.current_path = path

        index = self.proxy.mapFromSource(self.fs_model.index(str(path)))
        self.tree.setCurrentIndex(index)
        self.tree.scrollTo(index)

        self.tree.viewport().update()

        worker = SegmentWorker(path, image, self.settings.segmenter, self.settings.segmenter_params())
        worker.signals.finished.connect(self._on_segmented)
        worker.signals.failed.connect(self._on_segment_failed)

        self.pool.start(worker)


    def _finish_session(self) -> None:
        """
        Save work before leaving current image w/ a confirmation dialog for saving unannotated images
        """
        if self.session is None:
            return
        if self.session.annotated:
            if self.session.dirty:
                self._try_save(self.session)
        elif self.session.dirty or not self.session.mask_path.exists():
            message = (f"{self.session.image_path.name} has no annotations.\n"
                       f"Save an all-{self.session.classes[0].name} mask?")
            # This is in case the user resets an existing mask and then exits, so technically you can restart the app to undo a mask clear
            if self.session.mask_path.exists():
                message += "\nChoosing No keeps the existing mask on disk."
            answer = QMessageBox.question(
                self,
                "Unannotated Image",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._try_save(self.session)


    def _try_save(self, session: AnnotationSession) -> bool:
        """Try to save session mask and push up filesystem errors that otherwise die quietly"""
        try:
            session.save()
        except OSError as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save session {session.mask_path}:\n{e}")
            return False
        self.tree.viewport().update()
        return True


    def _show(self, image: np.ndarray) -> None:
        self._display_image = image
        if self._image_artist is None:
            self.axes.clear()
            self.axes.set_axis_off()
            self._image_artist = self.axes.imshow(image, interpolation="nearest")
            self.axes.set_aspect("equal", adjustable="datalim")
        else:
            self._update_viewport()
        self.canvas.draw_idle()


    def _update_viewport(self) -> None:
        """Give the artist only the visible region of the image"""
        if self._display_image is None or self._image_artist is None:
            return
        height, width = self._display_image.shape[:2]
        x0, x1 = self.axes.get_xlim()
        y1, y0 = self.axes.get_ylim()  # Inverted for images
        col0, col1 = max(int(x0), 0), min(int(x1) + 2, width)
        row0, row1 = max(int(y0), 0), min(int(y1) + 2, height)
        if col1 <= col0 or row1 <= row0:
            return  # No part of the image is visible
        step = max(1, round((col1 - col0) / self.axes.bbox.width), round((row1 - row0) / self.axes.bbox.height))
        view = self._display_image[row0:row1:step, col0:col1:step]
        self._image_artist.set_data(view)
        self._image_artist.set_extent((col0 - 0.5, col0 + view.shape[1] * step - 0.5,
                                      row0 + view.shape[0] * step - 0.5, row0 - 0.5))


    def _on_segmented(self, path: Path, image: np.ndarray, labels: np.ndarray) -> None:
        if path != self._pending_path:
            return  # Stale result
        self._pending_path = None
        session = AnnotationSession(path, image, labels)
        self.session = session
        self._show(self._render())
        if session.load_warning is not None:
            QMessageBox.warning(self,
                                "Mask Not Loaded",
                                session.load_warning + "\nSaving will overwrite the existing mask.")
        self.statusBar().showMessage(f"{path.name} - {int(labels.max()) + 1} segments")


    def _render(self) -> np.ndarray:
        assert self.session is not None  # Guarded against everywhere I call it, should never be an issue
        color = self.settings.boundary_color
        rgb = (int(color[1:3], 16),
               int(color[3:5], 16),
               int(color[5:7], 16))
        return self.session.render_display(boundary_color=rgb, boundaries=self.borders_action.isChecked())


    def _on_segment_failed(self, path: Path, error: str) -> None:
        if path != self._pending_path:
            return
        self._pending_path = None
        self.statusBar().showMessage(f"Segmentation failed - {path.name} - {error}")


    def _on_canvas_press(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.button == MouseButton.MIDDLE:
            self._pan_anchor = (event.xdata, event.ydata)
        elif event.button in (MouseButton.LEFT, MouseButton.RIGHT):
            class_index = 0 if event.button == MouseButton.RIGHT else self._current_class()
            self._paint(event.xdata, event.ydata, class_index)


    def _current_class(self) -> int:
        action = self.class_group.checkedAction()
        return action.data() if action else 0


    def _paint(self, x: float, y: float, class_index: int) -> None:
        if self.session is None:
            return
        column, row = int(round(x)), int(round(y))
        height, width = self.session.labels.shape
        if not (0 <= column < width and 0 <= row < height):
            return
        if self.session.assign(column, row, class_index):
            self._show(self._render())


    def _on_scroll(self, event: MouseEvent) -> None:
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return
        factor = 1 / 1.2 if event.step > 0 else 1.2
        x0, x1 = self.axes.get_xlim()
        y0, y1 = self.axes.get_ylim()
        self.axes.set_xlim(event.xdata + (x0 - event.xdata) * factor,
                           event.xdata + (x1 - event.xdata) * factor)
        self.axes.set_ylim(event.ydata + (y0 - event.ydata) * factor,
                           event.ydata + (y1 - event.ydata) * factor)
        self._update_viewport()
        self.canvas.draw_idle()


    def _on_motion(self, event: MouseEvent) -> None:
        if self._pan_anchor is None or event.xdata is None or event.ydata is None:
            return
        now = time.monotonic()  # For throttling panning to avoid event flooding
        if now - self._last_pan_draw < 0.03:
            return
        self._last_pan_draw = now
        dx = event.xdata - self._pan_anchor[0]
        dy = event.ydata - self._pan_anchor[1]
        x0, x1 = self.axes.get_xlim()
        y0, y1 = self.axes.get_ylim()
        self.axes.set_xlim(x0 - dx, x1 - dx)
        self.axes.set_ylim(y0 - dy, y1 - dy)
        self._update_viewport()
        self.canvas.draw_idle()


    def _on_release(self, event: MouseEvent) -> None:
        if event.button == MouseButton.MIDDLE:
            self._pan_anchor = None  # Reset the pan anchor

    def _clear_mask(self) -> None:
        """
        Clears the mask of any annotations by resetting the class id of every superpixel to 0
        """

        # No reason to reset a nonexistent session or a mask that doesn't have any annotations yet 
        if self.session is None or not self.session.annotated:
            self.statusBar().showMessage("Nothing to clear")
            return

        # Make the user confirm whether they really want to clear all the annotations, since a clear can't currently be undone
        answer = QMessageBox.question(
            self,
            "Clear Mask",
            f"Clear all annotations for {self.session.image_path.name}?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        # `and` short circuits so session.reset() only gets invoked if the answer is yes
        if answer == QMessageBox.StandardButton.Yes and self.session.reset():
            self._show(self._render())


    def _current_path(self) -> Path | None:
        """
        Returns either the session's path or _pending_path while segmentation is running - Gets the path to the "current" image
        """
        if self.session is not None:
            return self.session.image_path
        return self._pending_path


    def _undo(self) -> None:
        if self.session is not None and self.session.undo():
            self._show(self._render())


    def _redo(self) -> None:
        if self.session is not None and self.session.redo():
            self._show(self._render())


    def _save(self) -> None:
        if self.session is None:
            return
        if self._try_save(self.session):
            self.statusBar().showMessage(f"Saved {self.session.mask_path}")
        else:
            self.statusBar().showMessage(f"Could not save {self.session.mask_path}")


    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        before = (self.settings.segmenter, self.settings.params)
        dialog.apply()
        config.save(self.settings)
        if self.session is None:
            return
        if (self.settings.segmenter, self.settings.params) != before:
            self.open_image(self.session.image_path)  # For re-segmenting on changed settings
        else:
            self._show(self._render())  # Don't re-segment if we just change the color


    def _reset_view(self) -> None:
        if self._display_image is None:
            return
        height, width = self._display_image.shape[:2]
        self.axes.set_xlim(-0.5, width - 0.5)
        self.axes.set_ylim(height - 0.5, -0.5)  # Inverted for images
        self._update_viewport()
        self.canvas.draw_idle()


    def _show_progress(self) -> None:
        current = self._current_path()
        if current is None:
            self.statusBar().showMessage("No image open")
            return
        folder = current.parent
        done, total = count_annotated(folder, IMAGE_EXTENSIONS)
        QMessageBox.information(
            self,
            "Annotation Progress",
            f"{folder.name}: {done} of {total} images annotated ({done / total:.0%})",
        )


    def _step_image(self, offset: int) -> None:
        """
        Steps through the current image list based on the passed offset
        Stops at either end of the folder instead of wrapping
        """

        # Get the current path (either the open image or pending image)
        current = self._current_path()
        if current is None:
            self.statusBar().showMessage("No image open")
            return

        # Get the list of images sorted in the same order they should appear in the file tree
        images = sorted((p for p in current.parent.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=_natural_key)
        try:
            index = images.index(current)
        except ValueError:
            return  # Open image doesn't exist as expected (renamed, moved, deleted, etc.)
        target = index + offset

        # Stop at either end of the folder instead of wrapping
        # This is an opinionated decision, if you want to change it that is your prerogative 
        if not 0 <= target < len(images):
            self.statusBar().showMessage("Already at the " + ("last" if offset > 0 else "first") + " image")
            return
        self.open_image(images[target])
        

    def _toggle_borders(self) -> None:
        if self.session is not None:
            self._show(self._render())


    def closeEvent(self, event: QCloseEvent):
        self._finish_session()
        event.accept()


class ImageDirProxy(QSortFilterProxyModel):
    """Hide useless (no imgs or subdirectories) folders & tint file browser entries w/ annotations"""

    def __init__(self, fs_model: QFileSystemModel) -> None:
        super().__init__()
        self.fs_model = fs_model
        self.current_path: Path | None = None
        self.setSourceModel(fs_model)


    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if index.column() != 0:
            return super().data(index, role)
        path = self._image_path(index)
        if path is None:
            return super().data(index, role)
        if role == Qt.ItemDataRole.ForegroundRole:
            if mask_path(path).exists():
                return QBrush(ANNOTATED_COLOR)
        if role == Qt.ItemDataRole.FontRole and (path == self.current_path):
            font = QFont()
            font.setBold(True)
            return font

        return super().data(index, role)


    def _image_path(self, index: QModelIndex) -> Path | None:
        source_index = self.mapToSource(index)
        if self.fs_model.isDir(source_index):
            return None
        path = Path(self.fs_model.filePath(source_index))
        return path if path.suffix.lower() in IMAGE_EXTENSIONS else None


    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        index = self.fs_model.index(source_row, 0, source_parent)
        if not self.fs_model.isDir(index):
            # Only show files with valid extensions
            return Path(self.fs_model.fileName(index)).suffix.lower() in IMAGE_EXTENSIONS
        try:
            with os.scandir(self.fs_model.filePath(index)) as entries:
                # Hide any useless folders
                return any(e.is_dir() or Path(e.name).suffix.lower() in IMAGE_EXTENSIONS for e in entries)
        except OSError:
            return False


class SegmentWorker(QRunnable):
    """Runs a segmenter on a background thread and signals resulting labels"""


    class Signals(QObject):
        finished = Signal(Path, np.ndarray, np.ndarray)  # path, img, lbl
        failed = Signal(Path, str)


    def __init__(self, path: Path, image: np.ndarray, segmenter: str, params: segmentation.ParamValues) -> None:
        super().__init__()
        self.path = path
        self.image = image
        self.segmenter = segmenter
        self.params = params
        self.signals = self.Signals()


    @Slot()
    def run(self) -> None:
        try:
            labels = segmentation.run_segmenter(self.segmenter, self.image, self.params)
        except Exception as e:
            self.signals.failed.emit(self.path, str(e))
            return
        self.signals.finished.emit(self.path, self.image, labels)


class SettingsDialog(QDialog):
    """Edit segmenter choice and its parameters (determined by segmenter parameter metadata)"""


    def __init__(self, settings: config.Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Segmentation Settings")
        self.settings = settings
        self._values = {key: dict(params) for key, params in settings.params.items()}
        self._editors: dict[str, QSpinBox | QDoubleSpinBox] = {}
        self._form_key = settings.segmenter
        self._boundary_color = QColor(settings.boundary_color)
        self.color_button = QPushButton()
        self.color_button.clicked.connect(self._pick_color)
        self._update_swatch()

        self.combo = QComboBox()
        for key, seg in segmentation.REGISTRY.items():
            self.combo.addItem(seg.label, key)
        self.combo.setCurrentIndex(self.combo.findData(settings.segmenter))

        self.form = QFormLayout()

        options = QFormLayout()
        options.addRow("Boundary color", self.color_button)

        buttons = QDialogButtonBox()
        # Below is not strictly ideal Qt API, but it shuts up an incorrect inspector warning
        buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.combo)
        layout.addLayout(self.form)
        layout.addLayout(options)
        layout.addWidget(buttons)

        self._build_form()
        self.combo.currentIndexChanged.connect(self._on_segmenter_changed)


    def _update_swatch(self) -> None:
        self.color_button.setStyleSheet(f"background-color: {self._boundary_color.name()};")


    def _build_form(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self._editors.clear()
        seg = segmentation.REGISTRY[self._form_key]
        values = self._values[seg.key]
        for param in seg.params:
            low = param.minimum if param.minimum is not None else 0
            high = param.maximum if param.maximum is not None else 1e9
            editor: QSpinBox | QDoubleSpinBox
            if param.type is int:
                editor = QSpinBox()
                editor.setRange(int(low), int(high))
                editor.setValue(int(values[param.key]))
            else:
                editor = QDoubleSpinBox()
                editor.setRange(float(low), float(high))
                editor.setValue(float(values[param.key]))
            self._editors[param.key] = editor
            self.form.addRow(param.label, editor)


    def _pick_color(self) -> None:
        color = QColorDialog.getColor(self._boundary_color, self, "Boundary Color")
        if color.isValid():
            self._boundary_color = color
            self._update_swatch()


    def _on_segmenter_changed(self) -> None:
        self._stash()
        self._form_key = self.combo.currentData()
        self._build_form()


    def _stash(self) -> None:
        """Copy current editor values into the working copy"""
        for param_key, editor in self._editors.items():
            self._values[self._form_key][param_key] = editor.value()


    def apply(self) -> None:
        """Write edited values back to settings"""
        self._stash()
        self.settings.segmenter = self._form_key
        self.settings.params = self._values
        self.settings.boundary_color = self._boundary_color.name()
