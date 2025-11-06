@echo off
call C:\Users\Zain_\anaconda3\Scripts\activate.bat audio-notes-gpu
python "C:\Users\Zain_\Dropbox\Personal\Data Science Projects\audio-to-notes\windows_entry.py" --silent %*
