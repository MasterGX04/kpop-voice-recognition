import json
import os
import copy
from collections import deque

class HistoryManager:
    """
    Manages the undo/redo stack for the audio tester application.
    Caps the history length using a FIFO queue to prevent excessive memory and file bloat.
    Stores both committed labels and uncommitted start/end points.
    """
    def __init__(self, apply_callback, get_file_path_callback, max_history=50):
        """
        Initialize the history manager.
        :param apply_callback: Function to call in the main app to visually apply the state.
        :param get_file_path_callback: Function to retrieve the current history JSON file path.
        :param max_history: Maximum number of undo steps to remember.
        """
        self.undoStack = deque(maxlen=max_history)
        self.redoStack = deque(maxlen=max_history)
        self.apply_callback = apply_callback
        self.get_file_path_callback = get_file_path_callback
        self.max_history = max_history
        
    def appendHistoryToFile(self):
        """
        Writes the current capped history stack to the JSON file.
        Because self.undoStack is capped at max_history, this file will never bloat.
        """
        historyPath = self.get_file_path_callback()
        if not historyPath:
            return
            
        try:
            # Convert deque to list for JSON serialization
            history_list = list(self.undoStack)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(historyPath), exist_ok=True)
            
            with open(historyPath, "w") as f:
                json.dump(history_list, f, separators=(",", ":"))
        except Exception as e:
            print(f"Could not write history to file: {e}")
    
    def pushUndoState(self, labels, unsaved_starts, unsaved_ends, description=""):
        """
        Save the current labels AND all uncommitted start/end points into the undo stack.
        Clears the redo stack because a new action invalidates forward history.
        """
        snapshot = {
            "labels": copy.deepcopy(labels),
            "unsaved_starts": copy.deepcopy(unsaved_starts),
            "unsaved_ends": copy.deepcopy(unsaved_ends), 
            "description": description,
        }
        self.undoStack.append(snapshot)
        self.redoStack.clear()
        self.appendHistoryToFile()

    def undo(self, current_labels, current_unsaved_starts, current_unsaved_ends):
        """
        Reverts to the previous state in the undo stack.
        Saves the current state to the redo stack before applying the undo so we can go forward again.
        """
        if not self.undoStack:
            print("Nothing to undo.")
            return

        # 1. Save current state into redo stack
        current_state = {
            "labels": copy.deepcopy(current_labels),
            "unsaved_starts": copy.deepcopy(current_unsaved_starts),
            "unsaved_ends": copy.deepcopy(current_unsaved_ends),
            "description": "auto-redo-snapshot",
        }
        self.redoStack.append(current_state)
        
        # 2. Pop the last state from undo stack and apply it
        state = self.undoStack.pop()
        self.apply_callback(state)
        self.appendHistoryToFile()
        print("Undo:", state.get("description", ""))

    def redo(self, current_labels, current_unsaved_starts, current_unsaved_ends):
        """
        Advances to the next state in the redo stack.
        Saves the current state to the undo stack before applying the redo.
        """
        if not self.redoStack:
            print("Nothing to redo.")
            return
        
        # 1. Save current state into undo stack
        current_state = {
            "labels": copy.deepcopy(current_labels),
            "unsaved_starts": copy.deepcopy(current_unsaved_starts),
            "unsaved_ends": copy.deepcopy(current_unsaved_ends),
            "description": "auto-undo-snapshot",
        }
        self.undoStack.append(current_state)
        
        # 2. Pop the last state from redo stack and apply it
        state = self.redoStack.pop()
        self.apply_callback(state)
        self.appendHistoryToFile()
        print("Redo:", state.get("description", ""))
        
    def clear(self):
        """
        Empties the history. Useful when loading a completely new song.
        """
        self.undoStack.clear()
        self.redoStack.clear()