# Simple Annotator

This repository contains a tool I created to solve a couple of headaches with the previous tool I was using to annotate images for the CIWA+ project. There's also a couple of nice-to-haves that I threw in since they were relatively light and, in my opinion, made for a better experience.

Basically, this was built from the ground up, using more modern options where available, for one specific purpose. It's not a tool to do a ton of different things; it is a tool for annotating. As such, the entire experience has been streamlined in order to make that specific task as simple as possible.

## Getting Started

**Requires Python 3.12+**

### 1. Clone and enter the project

```bash
git clone https://github.com/qdepfyffer/simple_annotator
cd simple_annotator
```

### 2. Create and activate a virtual environment

**Linux/MacOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PS):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell refuses to run the activation script, run
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then retry. *
> 
> \* https://stackoverflow.com/a/4038991

**Windows (CMD):**

```bat
py -m venv .venv
.venv\Scripts\Activate.bat
```

### 3. Install

```bash
pip install -e .
```

### 4. Run

```bash
simple-annotator
```

(Alternatively: 'python -m simple_annotator')

## Configuration

Settings (selected segmentation algorithm and parameters) can be edited in the settings panel, accessible via the 'Settings' button on the toolbar. Settings are saved automatically and persist between runs.

The config file should live at:

| OS      | Path                                                             |
|---------|------------------------------------------------------------------|
| Linux   | `~/.config/simple_annotator/config.json` *                       |
| macOS   | `~/Library/Application Support/simple_annotator/config.json`     |
| Windows | `C:\Users\<username>\AppData\Local\simple_annotator\config.json` |

\* respects `$XDG_CONFIG_HOME` if set.

Changing the settings mid-annotation will re-segment the image and attempt to rebuild the mask with the new settings. Since the mask gets rebuilt on a change of settings, the undo / redo queues end up being cleared as well. This is intended behavior.

Deleting the file resets all settings to defaults.

## What's different

The old tool we were using had a few glaring bugs:

* You could only annotate one image before having to close the tool and re-open it to annotate the next one.
* If your source image had any extension besides `.jpg`, the resulting mask would not be guaranteed to save as a `.png`, which lead to some severe mask quality issues.

Presumably, those two bugs were responsible for other broken behavior as well. This tool fixes both of those issues (so far as I have tested). I experienced a number of pain points when using the old tool, so decided to change a few other things:

1. Added undo and redo functionality!
2. With the old tool you had to house your images to be annotated in a folder named 'train'. Now you do not. It will append '_labels' to the name of whatever folder holds your images to be annotated and create the masks there. Otherwise, the old tool would create the masks in the same directory that held the images to be annotated.
   * I debated calling this a bug, but looking at the code for the old tool, it seems to be intended behavior. 
     * see: https://github.com/AIISLab/Sunlit-leaf-annotator/blob/main/src/pynovisao.py#L192C2-L193
3. Opening a source image no longer overwrites any potentially existing mask. Instead, simple_annotator will rebuild your progress from the existing mask. This means if you want to go back and improve a mask later, you won't have to start from scratch.
4. The file browser is much more integrated into the UI. This cuts down on repetitive clicks between images, especially since we no longer have to close and reopen the application whenever we want to annotate a new image. 
5. Far more accessible settings. Before, users would have had to dig through the code to change any settings. Now, there is a settings panel accessible through the toolbar at the top that allows users to directly edit and save the settings for a given segmenter. This is covered more in depth in the 'Configuration' section.
6. You can no longer color the same superpixel multiple times and superpixels that are manually annotated as noise (or whatever default class you implement) now remain uncolored, which leaves a cleaner visual interface.
7. This tool **does not** write to disk on every click. Could this lose you progress? Maybe. Does it cut the number of disk writes the tool does? Yeah, potentially by a ridiculous amount. Instead, saving is done either manually (via the save button or your platform's default 'Save' key command) or on switching images / closing the tool.
8. This tool **does not** immediately create a mask for an opened image. It only creates a mask once the first superpixel is colored. This means if you accidentally open an image it won't create unwanted images. This comes with an added dialog when switching away from the current image or closing the tool when no annotations have been made for the current image. This dialog asks if the user really wants to save a mask with zero annotations. 
9. This tool **does** immediately segment the image using the preconfigured segmenter specified in the config. The user no longer has to manually run the segmentation via a menu.

## What's to come 

*(I'll cross these off as they're added or something)*

* Zooming, if possible, to help with annotations
* Easier between-image navigation to further help speed up annotation
* A button to reset the mask
* Caching of the superpixel border overlay
* Setting for superpixel border overlay color
* Better scaling of superpixel border overlay on smaller images
* ~~A button to audit annotation progress (how many labels have been completed)~~
* Other segmentation algorithms
* Mystery features I haven't thought of yet

## Note

None of this is meant as a dig at the old tool. It still worked for what we needed it for. However, maintenance of said tool seems to have fallen by the wayside as contributors have come and gone, and I felt like it was time for something more efficient and modern. Also, I needed something to do. 
