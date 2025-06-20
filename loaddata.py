from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QTableWidget,
    QTableWidgetItem, QFrame, QCheckBox, QMessageBox, QLineEdit, QGroupBox,
    QDialog, QRadioButton, QListWidgetItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QPoint
from PyQt6.QtGui import QIcon, QColor, QCursor, QPainter
import polars as pl
import os
from typing import List, Dict, Any, Optional
import sys

# Custom color scheme
COLORS = {
    "primary": "#006A71",    # Deep teal - for main actions
    "primary_dark": "#005a61",  # Darker teal
    "secondary": "#4F959D",  # Light teal - for secondary actions
    "secondary_dark": "#3f858d",  # Darker light teal
    "accent": "#98D2C0",     # Mint green - for accents
    "accent_dark": "#88c2b0",  # Darker mint green
    "background": "#F6F8D5", # Light cream background
    "surface": "#ffffff",    # White surface
    "text": "#006A71",      # Deep teal text
    "text_light": "#F6F8D5", # Light cream text for dark backgrounds
    "text_secondary": "#4F959D", # Light teal secondary text
    "success": "#4F959D",   # Light teal - for success states
    "error": "#FF6B6B",     # Bright red - for errors
    "warning": "#FFB86B",   # Warm orange - for warnings
    "danger": "#FF6B6B",    # Deep red - for destructive actions
    "danger_dark": "#e55f5f", # Darker red
    "sidebar": "#006A71",   # Deep teal sidebar
    "card": "#ffffff",      # White card
}

COMBINED_TEMP_KEY = "__COMBINED_TEMP__"
COMBINED_TEMP_LABEL = "Combined (Temporary)"

class ToggleSwitch(QCheckBox):
    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self.setFixedSize(45, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # It's good practice to trigger a repaint when the state that affects painting changes
        self.stateChanged.connect(self.update)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        rect = self.contentsRect()
        
        # Define colors from the global theme
        bg_color_off = QColor("#888")
        bg_color_on = QColor(COLORS["primary"])
        circle_color = QColor(COLORS["surface"])

        # Draw background track
        p.setBrush(bg_color_on if self.isChecked() else bg_color_off)
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        # Draw handle
        p.setBrush(circle_color)
        padding = 2
        handle_radius = (rect.height() / 2) - padding
        
        # Calculate handle position
        start_pos_x = padding + handle_radius
        end_pos_x = self.width() - padding - handle_radius
        current_pos_x = end_pos_x if self.isChecked() else start_pos_x
        center_y = rect.height() / 2
        
        p.drawEllipse(QPoint(int(current_pos_x), int(center_y)), int(handle_radius), int(handle_radius))

class FileListItemWidget(QWidget):
    delete_clicked = pyqtSignal(str)
    def __init__(self, file_label, file_key, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)
        self.label = QLabel(file_label)
        layout.addWidget(self.label)
        self.delete_btn = QPushButton("❌")
        self.delete_btn.setFixedSize(16, 16)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                border: none;
                font-size: 9px;
                border-radius: 6px;
                background: transparent;
                padding: 2px 4px;
            }
            QPushButton:hover {
                background: #FF6B6B;
                color: white;
                border-radius: 2px;
            }
        """)
        self.delete_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        layout.addWidget(self.delete_btn)
        self.delete_btn.clicked.connect(lambda: self.delete_clicked.emit(file_key))

# Worker Thread for data processing
class DataLoadWorker(QThread):
    data_loaded_signal = pyqtSignal(pl.DataFrame, str) # Signal to send processed DataFrame and original file path
    error_signal = pyqtSignal(str) # Signal for errors

    def __init__(self, file_path: str, process_file_func):
        super().__init__()
        self.file_path = file_path
        self.process_file_func = process_file_func

    def run(self):
        try:
            df = self.process_file_func(self.file_path)
            self.data_loaded_signal.emit(df, self.file_path)
        except Exception as e:
            self.error_signal.emit(f"Error processing file '{os.path.basename(self.file_path)}': {str(e)}")

class DataProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Processing")
        self.setGeometry(100, 100, 800, 600) # x, y, width, height

        # Data storage
        self.selected_files: List[str] = []
        self.combined_df: Optional[pl.DataFrame] = None
        self.remove_duplicates_var = True # Using a simple boolean, default True
        
        # Template mapping storage
        self.template_file: Optional[str] = None
        self.header_mapping: Dict[str, str] = {}
        self.filename_mapping: Dict[str, Dict[str, str]] = {} # Initialize filename_mapping
        self.columns_to_delete: List[str] = []
        
        # Data preview state
        self.current_preview_df: Optional[pl.DataFrame] = None
        self.original_file_df_for_preview: Optional[pl.DataFrame] = None
        self.changed_header_df_for_preview: Optional[pl.DataFrame] = None
        self.preview_mode_var = "original_file"  # Using a string now
        self.last_displayed_file_path: Optional[str] = None

        # Add last directory tracking
        self.last_directory = os.path.expanduser("~")  # Default to user's home directory

        # Caching and Debouncing
        self.dataframe_cache: Dict[str, pl.DataFrame] = {}
        self.load_delay_timer = QTimer(self)
        self.load_delay_timer.setSingleShot(True)
        self.load_delay_timer.timeout.connect(self._start_loading_file)
        self.pending_load_path: Optional[str] = None

        # Create main layout
        self._create_main_layout()
        
        # Initialize button states
        self.update_button_states()

        # Initialize worker thread and signals
        self.data_worker = None # Will be initialized in on_file_selected_for_display

        self.loading_gif = None  # Will be initialized in create_main_content
        self.loading_gif_label = None
        self.loading_frame = None

    def _create_main_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove outer margin
        main_layout.setSpacing(0)  # Remove spacing between sidebar and content

        # Create sidebar and main content
        self.create_sidebar(main_layout)
        self.create_main_content(main_layout)

    def create_sidebar(self, main_layout: QHBoxLayout):
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet(f"background-color: {COLORS['sidebar']}; border-radius: 0px;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        sidebar_layout.setContentsMargins(16, 20, 16, 20)  # Add padding: left, top, right, bottom

        # App title
        app_title = QLabel("Data Processing")
        app_title.setStyleSheet(f"color: {COLORS['text_light']}; font-size: 20px; font-weight: bold;")
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(app_title)

        # Action buttons
        button_style = f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: {COLORS['text_light']};
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border: 2px solid {COLORS['secondary_dark']};
                margin: 5px 0px; /* Add vertical margin */
                min-width: 140px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary_dark']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_dark']};
            }}
        """

        select_button = QPushButton("Select Files")
        select_button.clicked.connect(self.select_files)
        select_button.setStyleSheet(button_style)
        select_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        select_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sidebar_layout.addWidget(select_button)

        combine_button = QPushButton("Combine Files")
        combine_button.clicked.connect(self.combine_all)
        combine_button.setStyleSheet(button_style)
        combine_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        combine_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sidebar_layout.addWidget(combine_button)

        delete_column_button = QPushButton("Delete Columns")
        delete_column_button.clicked.connect(self.delete_columns)
        delete_column_button.setStyleSheet(button_style)
        delete_column_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        delete_column_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sidebar_layout.addWidget(delete_column_button)

        # Move the status section to the bottom of the sidebar
        sidebar_layout.addStretch() # Push status to bottom
        # Status section
        status_frame = QFrame()
        status_frame.setStyleSheet("background-color: transparent;")
        status_layout = QVBoxLayout(status_frame)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet(f"color: {COLORS['text_light']}; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)
        sidebar_layout.addWidget(status_frame)

        main_layout.addWidget(sidebar)

    def create_main_content(self, main_layout: QHBoxLayout):
        content_widget = QWidget()
        content_widget.setStyleSheet(f"background-color: {COLORS['surface']};")  # Change to white background
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(4, 10, 4, 4)  # Match sidebar padding
        content_layout.setSpacing(0)  # Add spacing between sections

        # Header section
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: transparent;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.addWidget(header_frame)

        # Title and subtitle
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: transparent;")
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(title_frame)

        title_label = QLabel("Selected Files")
        title_label.setStyleSheet(f"color: {COLORS['primary']}; font-size: 18px; font-weight: bold;")
        title_layout.addWidget(title_label)

        subtitle_label = QLabel("Select and combine your data files")
        subtitle_label.setStyleSheet("color: gray; font-size: 12px;")
        title_layout.addWidget(subtitle_label)
        
        header_layout.addStretch()

        # Buttons frame
        buttons_frame = QFrame()
        buttons_frame.setStyleSheet("background-color: transparent;")
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(5)
        header_layout.addWidget(buttons_frame)
        
        button_style_header = f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
                padding: 8px 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: #f2f2f2;
                color: #888;
                border: none;
            }}
            QPushButton:pressed {{
                background-color: {COLORS['secondary_dark']};
            }}
        """
        
        # Select Template button
        self.select_template_btn = QPushButton("Select Template")
        self.select_template_btn.clicked.connect(self.select_template)
        self.select_template_btn.setStyleSheet(button_style_header)
        self.select_template_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        buttons_layout.addWidget(self.select_template_btn)
        
        # Change Header button
        self.change_header_btn = QPushButton("Change Header")
        self.change_header_btn.clicked.connect(self.change_header)
        self.change_header_btn.setStyleSheet(button_style_header)
        self.change_header_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        buttons_layout.addWidget(self.change_header_btn)

        # Export button
        self.export_btn = QPushButton("Export File")
        self.export_btn.clicked.connect(self.export_file)
        self.export_btn.setStyleSheet(button_style_header)
        self.export_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        buttons_layout.addWidget(self.export_btn)

        # File list section
        file_list_container = QFrame()
        file_list_container.setStyleSheet("background-color: transparent;")
        file_list_layout = QVBoxLayout(file_list_container)
        file_list_layout.setContentsMargins(15, 0, 15, 15)
        file_list_layout.setSpacing(5)
        content_layout.addWidget(file_list_container)

        # Remove duplicates checkbox and Clear all (button) in the same row at the top of the file list section
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(0)
        self.remove_duplicates_checkbox = QCheckBox("Remove duplicate rows when combining files")
        self.remove_duplicates_checkbox.setChecked(True)
        self.remove_duplicates_checkbox.setStyleSheet(f"color: {COLORS['text']}; font-size: 11px;")
        self.remove_duplicates_checkbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        top_row.addWidget(self.remove_duplicates_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)
        top_row.addStretch()
        self.clear_all_btn = QPushButton("Clear all")
        self.clear_all_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.clear_all_btn.setEnabled(False)
        self.clear_all_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['danger']};
                color: white;
                font-size: 11px;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
                min-width: 48px;
                max-width: 80px;
            }}
            QPushButton:disabled {{
                background: #e0e0e0;
                color: #888;
            }}
            QPushButton:hover:!disabled {{
                background: #ff8787;
                color: #fff;
            }}
        """)
        self.clear_all_btn.clicked.connect(self.clear_files)
        top_row.addWidget(self.clear_all_btn, alignment=Qt.AlignmentFlag.AlignRight)
        file_list_layout.insertLayout(0, top_row)

        # File list (QListWidget)
        self.listbox = QListWidget()
        self.listbox.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['surface']};
                color: {COLORS['text']};
                font-size: 11px;
                border: 1px solid {COLORS['primary']};
                border-radius: 5px;
                padding: 5px;
            }}
            QListWidget::item {{
                padding: 3px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
            QScrollBar:vertical {{
                background: #f0f0f0;
                width: 10px;
                margin: 2px 0 2px 0;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['primary']};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['secondary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)
        self.listbox.itemClicked.connect(self.on_file_selected_for_display)
        file_list_layout.addWidget(self.listbox)

        # Data Preview Section
        self.data_preview_frame = QFrame()
        self.data_preview_frame.setStyleSheet("background-color: transparent;")
        data_preview_layout = QVBoxLayout(self.data_preview_frame)
        data_preview_layout.setContentsMargins(15, 0, 15, 15)
        data_preview_layout.setSpacing(5)
        content_layout.addWidget(self.data_preview_frame)

        # Preview header
        preview_header_layout = QHBoxLayout()
        preview_header_layout.setContentsMargins(0, 0, 0, 0)
        preview_header_layout.setSpacing(10)
        data_preview_layout.addLayout(preview_header_layout)

        # Left side: Preview label
        preview_label = QLabel("Data Preview")
        preview_label.setStyleSheet(f"color: {COLORS['primary']}; font-size: 14px; font-weight: bold;")
        preview_header_layout.addWidget(preview_label)

        # Data table
        self.data_table = QTableWidget()
        self.data_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['primary']};
                border-radius: 5px;
                gridline-color: #eee;
                font-size: 10px;
                selection-background-color: {COLORS['primary']};
                selection-color: white;
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QTableWidget::item:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
            QHeaderView::section {{
                background-color: {COLORS['primary_dark']};
                color: white;
                padding: 4px;
                border: 1px solid #777;
                font-weight: bold;
                font-size: 10px;
            }}
            QScrollBar:vertical {{
                background: #f0f0f0;
                width: 10px;
                margin: 2px 0 2px 0;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['primary']};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['secondary']};
            }}
            QScrollBar:horizontal {{
                background: #f0f0f0;
                height: 10px;
                margin: 0 2px 0 2px;
                border-radius: 5px;
            }}
            QScrollBar::handle:horizontal {{
                background: {COLORS['primary']};
                min-width: 20px;
                border-radius: 5px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {COLORS['secondary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                height: 0px; width: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background: none;
            }}
        """)
        self.data_table.setSortingEnabled(True)
        self.data_table.setCornerButtonEnabled(False)
        data_preview_layout.addWidget(self.data_table)
        
        main_layout.addWidget(content_widget)

    def update_button_states(self):
        """Update the state of action buttons based on whether files are selected and template is loaded."""
        has_files = len(self.selected_files) > 0
        has_template = self.template_file is not None and (self.header_mapping or (hasattr(self, 'filename_mapping') and self.filename_mapping))
        has_combined_data = self.combined_df is not None
        if has_files:
            self.export_btn.setEnabled(True)
            self.select_template_btn.setEnabled(True)
            if hasattr(self, 'clear_all_btn'):
                self.clear_all_btn.setEnabled(True)
            if has_template:
                self.change_header_btn.setEnabled(True)
            else:
                self.change_header_btn.setEnabled(False)
        else:
            self.select_template_btn.setEnabled(False)
            self.change_header_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            if hasattr(self, 'clear_all_btn'):
                self.clear_all_btn.setEnabled(False)

    def select_files(self):
        """Opens file dialog and adds selected files to the listbox, appending only new files."""
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select Files", self.last_directory,  # Use last_directory
                "All supported files (*.parquet *.csv *.xlsx *.xls);;Parquet files (*.parquet);;CSV files (*.csv);;Excel files (*.xlsx *.xls);;All files (*.*)"
            )

            if files:
                self.last_directory = os.path.dirname(files[0])
                # Add only new files (no duplicates)
                new_files = [f for f in files if f not in self.selected_files and f != COMBINED_TEMP_KEY]
                if new_files:
                    self.selected_files.extend(new_files)
                    self.update_listbox()  # Use custom widget with delete button
                    self.update_status(f"Added {len(new_files)} new file(s)")
                    self.combined_df = None
                    self.update_button_states()
                else:
                    self.update_status("No new files were added")

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error during file selection: {str(e)}"
            )
            self.update_status("Error selecting files")

    def update_listbox(self):
        self.listbox.clear()
        for file in self.selected_files:
            if file == COMBINED_TEMP_KEY:
                label = COMBINED_TEMP_LABEL
            else:
                label = os.path.basename(file)
            item_widget = FileListItemWidget(label, file)
            item = QListWidgetItem(self.listbox)
            item.setSizeHint(QSize(0, 28))
            item.setData(Qt.ItemDataRole.UserRole, file)
            self.listbox.addItem(item)
            self.listbox.setItemWidget(item, item_widget)
            item_widget.delete_clicked.connect(self._remove_file_from_list)

    def _remove_file_from_list(self, file_key):
        if file_key in self.selected_files:
            self.selected_files.remove(file_key)
            self.update_listbox()
            self.update_button_states()
            # If the removed file is currently displayed, clear preview
            if self.last_displayed_file_path == file_key:
                self.data_table.clearContents()
                self.data_table.setRowCount(0)
                self.data_table.setColumnCount(0)
                self.update_status("File removed from list.")

    def clear_files(self):
        """Clears all selected files."""
        self.selected_files = []
        if self.listbox:
            self.listbox.clear()
            # self.listbox.update()  # Removed to reduce flicker
        self.combined_df = None
        self.update_status("Cleared all files")
        self.update_button_states()
        # Clear the data preview table as well
        if hasattr(self, 'data_table') and self.data_table is not None:
            self.data_table.clearContents()
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
        # self.update()  # Removed to reduce flicker

    def export_file(self):
        """Export the processed files with various options."""
        if not self.selected_files:
            QMessageBox.warning(
                self, "Warning",
                "Please select files first before exporting!"
            )
            return
            
        try:
            # Create export options dialog
            export_dialog = QDialog(self)
            export_dialog.setWindowTitle("Export Options")
            export_dialog.setModal(True)

            dialog_layout = QVBoxLayout(export_dialog)

            # Export mode selection
            mode_group_box = QGroupBox("Export Mode")
            mode_layout = QVBoxLayout(mode_group_box)
            
            self.export_mode_combined_radio = QRadioButton("Export as single combined file")
            self.export_mode_combined_radio.setChecked(True)
            self.export_mode_separate_radio = QRadioButton("Export as separate files")
            
            mode_layout.addWidget(self.export_mode_combined_radio)
            mode_layout.addWidget(self.export_mode_separate_radio)
            dialog_layout.addWidget(mode_group_box)

            # File naming options
            naming_group_box = QGroupBox("File Naming")
            naming_layout = QVBoxLayout(naming_group_box)

            self.naming_mode_original_radio = QRadioButton("Keep original filenames")
            self.naming_mode_original_radio.setChecked(True)
            self.naming_mode_prefix_radio = QRadioButton("Add prefix to filenames")

            self.prefix_entry = QLineEdit()
            self.prefix_entry.setPlaceholderText("Enter prefix (e.g., 'processed_')")
            self.prefix_entry.setEnabled(False) # Initially disabled

            # Connect radio buttons to enable/disable prefix entry
            self.naming_mode_prefix_radio.toggled.connect(self.prefix_entry.setEnabled)

            naming_layout.addWidget(self.naming_mode_original_radio)
            naming_layout.addWidget(self.naming_mode_prefix_radio)
            naming_layout.addWidget(self.prefix_entry)
            dialog_layout.addWidget(naming_group_box)

            # Export button
            export_btn = QPushButton("Export")
            export_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['primary']};
                    color: white;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 8px 12px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS['primary_dark']};
                }}
                QPushButton:pressed {{
                    background-color: {COLORS['secondary_dark']};
                }}
            """)
            export_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            dialog_layout.addWidget(export_btn)
            export_btn.clicked.connect(export_dialog.accept)

            # Execute dialog and get result
            result = export_dialog.exec()

            if result == QDialog.DialogCode.Accepted:
                mode = "combined" if self.export_mode_combined_radio.isChecked() else "separate"
                naming = "original" if self.naming_mode_original_radio.isChecked() else "prefix"
                prefix = self.prefix_entry.text().strip() if naming == "prefix" else ""

                if mode == "combined":
                    self._export_combined_file()
                else:
                    self._export_separate_files(naming, prefix)

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error preparing export: {str(e)}"
            )
            self.update_status("Export Failed")

    def _export_combined_file(self):
        """Export all files as a single combined file."""
        if self.combined_df is None:
            QMessageBox.warning(
                self, "Warning",
                "Please combine files first using the 'Combine Files' button before exporting a combined file!"
            )
            return

        try:
            save_path = self._get_save_path()
            if not save_path:
                return

            self._save_dataframe(save_path)

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error exporting combined file: {str(e)}"
            )
            self.update_status("Export Failed")

    def _export_separate_files(self, naming_mode: str, prefix: str = ""):
        """Export files separately with specified naming convention."""
        try:
            # Get save directory
            save_dir = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Files"
            )
            if not save_dir:
                return

            # Process each file
            success_count = 0
            error_files = []

            for file_path in self.selected_files:
                try:
                    # Get original filename and extension
                    original_name = os.path.basename(file_path)
                    name, ext = os.path.splitext(original_name)

                    # Create new filename based on naming mode
                    if naming_mode == "prefix":
                        new_name = f"{prefix}{name}{ext}"
                    else:
                        new_name = original_name

                    # Create full save path
                    save_path = os.path.join(save_dir, new_name)

                    # Process and save the file
                    df = self.process_file(file_path)
                    if not df.is_empty():
                        self._save_single_file(df, save_path)
                        success_count += 1
                    else:
                        error_files.append(f"{original_name} (empty file)")

                except Exception as e:
                    error_files.append(f"{original_name} ({str(e)})")

            # Show results
            if success_count > 0:
                success_msg = f"Successfully exported {success_count} files to:\n{save_dir}"
                if error_files:
                    success_msg += "\n\nFailed files:\n" + "\n".join(f"- {f}" for f in error_files)
                QMessageBox.information(self, "Export Complete", success_msg)
                self.update_status(f"Exported {success_count} files")
            else:
                QMessageBox.critical(
                    self, "Export Failed",
                    "No files were successfully exported.\n\nFailed files:\n" +
                    "\n".join(f"- {f}" for f in error_files)
                )
                self.update_status("Export Failed")

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error during separate file export: {str(e)}"
            )
            self.update_status("Export Failed")

    def _save_single_file(self, df: pl.DataFrame, save_path: str):
        """Save a single DataFrame to the specified path."""
        ext = os.path.splitext(save_path)[1].lower()

        # Remove double quotes from string columns before saving
        string_columns = [col for col in df.columns if df.schema[col] == pl.String]
        if string_columns:
            df = df.with_columns([
                pl.col(col).str.replace('"', '') for col in string_columns
            ])

        if ext == '.parquet':
            df.write_parquet(save_path)
        elif ext == '.csv':
            df.write_csv(save_path)
        elif ext == '.xlsx':
            try:
                df.write_excel(save_path)
            except ImportError:
                raise ImportError("Exporting to Excel requires 'openpyxl'. Please install it:\npip install openpyxl")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def change_header(self):
        """
        Change column headers using the previously loaded header mapping template.
        This function applies the mappings stored in self.header_mapping to rename columns.
        If filename-specific mappings exist, they will be applied only to matching files.
        """
        # Check if a file is selected
        if not self.selected_files:
            QMessageBox.warning(
                self, "Warning",
                "Please select a file first before changing headers!"
            )
            return
        
        # Check if a template has been loaded
        if not self.template_file or (not self.header_mapping and not hasattr(self, 'filename_mapping')):
            QMessageBox.warning(
                self, "Warning",
                "Please select a template file first using the 'Select Template' button!"
            )
            return
            
        try:
            # Process each file
            processed_files = []
            skipped_files = []
            total_changes = 0
            self.processed_dfs_with_changed_headers = {}  # Dict[file_path, DataFrame]
            
            for source_file in self.selected_files:
                source_file_name = os.path.basename(source_file)
                
                # Get the appropriate mapping for this file
                mapping = self.header_mapping # Start with general mapping
                if hasattr(self, 'filename_mapping'):
                    found_specific_mapping = False
                    # Check for exact match first
                    if source_file_name in self.filename_mapping:
                        mapping = self.filename_mapping[source_file_name]
                        found_specific_mapping = True
                    else:
                        # Check if any filename in the mapping is a substring of the source file name
                        for template_filename, specific_mapping in self.filename_mapping.items():
                            if template_filename in source_file_name:
                                mapping = specific_mapping
                                found_specific_mapping = True
                                break
                if not mapping:
                    skipped_files.append(source_file_name)
                    continue
                try:
                    df = self.process_file(source_file)
                    if df.is_empty():
                        QMessageBox.critical(
                            self, "Error",
                            f"The source file '{source_file_name}' resulted in an empty DataFrame!"
                        )
                        continue
                except Exception as e:
                    QMessageBox.critical(
                        self, "Processing Error",
                        f"Error processing file '{source_file_name}':\n\n{str(e)}"
                    )
                    self.update_status(f"Processing failed: {source_file_name}")
                    continue
                old_columns = df.columns
                new_columns = []
                seen_names = {}
                for old_col in old_columns:
                    new_col = mapping.get(old_col, old_col)
                    if new_col in seen_names:
                        seen_names[new_col] += 1
                        new_col = f"{new_col}_{seen_names[new_col]}"
                    else:
                        seen_names[new_col] = 0
                    new_columns.append(new_col)
                rename_mapping = dict(zip(old_columns, new_columns))
                renamed_df = df.rename(rename_mapping)
                self.processed_dfs_with_changed_headers[source_file] = renamed_df
                changes = sum(1 for old, new in zip(old_columns, new_columns) if old != new)
                total_changes += changes
                if changes > 0:
                    processed_files.append((source_file_name, changes))
            success_msg = ["Headers successfully changed!"]
            if processed_files:
                success_msg.append("\nProcessed files:")
                for filename, changes in processed_files:
                    success_msg.append(f"- {filename}: {changes} columns changed")
            if skipped_files:
                success_msg.append("\nSkipped files (no applicable mapping):")
                for filename in skipped_files:
                    success_msg.append(f"- {filename}")
            success_msg.append(f"\nTotal columns changed: {total_changes}")
            QMessageBox.information(
                self, "Success",
                "\n".join(success_msg)
            )
            self.update_status(f"Changed headers in {len(processed_files)} files")
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error changing headers: {str(e)}"
            )
            self.update_status("Header change failed")

    def _get_save_path(self) -> Optional[str]:
        """Get the save path from file dialog."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", "combined_data",
            "CSV files (*.csv);;Parquet files (*.parquet);;Excel files (*.xlsx);;All files (*.*)"
        )
        return file_path

    def _save_dataframe(self, save_path: str):
        """Save DataFrame to the specified path."""
        ext = os.path.splitext(save_path)[1].lower()

        self.update_status(f"Exporting to {save_path}...")
        print(f"Exporting combined data to {save_path}")

        # Remove double quotes from string columns before saving
        string_columns = [col for col in self.combined_df.columns if self.combined_df.schema[col] == pl.String]
        if string_columns:
            self.combined_df = self.combined_df.with_columns([
                pl.col(col).str.replace('"', '') for col in string_columns
            ])

        if ext == '.parquet':
            self.combined_df.write_parquet(save_path)
        elif ext == '.csv':
            self.combined_df.write_csv(save_path)
        elif ext == '.xlsx':
            try:
                self.combined_df.write_excel(save_path)
            except ImportError:
                QMessageBox.critical(
                    self, "Export Error",
                    "Exporting to Excel requires 'openpyxl'. Please install it:\n\npip install openpyxl"
                )
                self.update_status("Export Failed (Excel: openpyxl missing)")
                return
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        QMessageBox.information(
            self, "Success",
            f"Combined data saved successfully to {save_path}!"
        )
        self.update_status("Export: Success")

    def update_status(self, message: str):
        """Updates the status label text with color coding."""
        if self.status_label:
            # Determine message type and color
            color = COLORS["text"]
            if "error" in message.lower() or "failed" in message.lower():
                color = COLORS["error"]
            elif "success" in message.lower():
                color = COLORS["success"]
            elif "warning" in message.lower():
                color = COLORS["warning"]

            self.status_label.setText(f"Status: {message}")
            self.status_label.setStyleSheet(f"color: {color};")
            # self.update()  # Removed to reduce flicker

    def process_file(self, file_path: str) -> pl.DataFrame:
        """
        Read and process a file using Polars optimized operations.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            pl.DataFrame: Processed DataFrame
            
        Raises:
            ValueError: If file type is unsupported or processing fails
        """
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)

        self.update_status(f"Processing: {file_name}")

        try:
            if ext == '.parquet':
                df = pl.read_parquet(file_path)
                # Remove double quotes from string columns
                string_columns = [col for col in df.columns if df.schema[col] == pl.String]
                if string_columns:
                    df = df.with_columns([
                        pl.col(col).str.replace('"', '') for col in string_columns
                    ])
                return df

            elif ext in ['.csv', '.xlsx', '.xls']:
                # Find header row
                header_row = find_header_start_row(file_path, ext)
                
                if ext == '.csv':
                    df = self._process_csv_file(file_path, header_row)
                else:
                    try:
                        # For Excel files, use fastexcel
                        df = pl.read_excel(
                            file_path, 
                            offset=header_row
                        )
                    except Exception as excel_err:
                        # Fallback to using openpyxl if fastexcel fails
                        try:
                            df = pl.read_excel(
                                file_path,
                                engine="openpyxl"
                            )
                        except Exception:
                            # Last resort: just read the file without options
                            df = pl.read_excel(file_path)
                
                # Remove double quotes from string columns
                string_columns = [col for col in df.columns if df.schema[col] == pl.String]
                if string_columns:
                    df = df.with_columns([
                        pl.col(col).str.replace('"', '') for col in string_columns
                    ])
                return df

            else:
                raise ValueError(f"Unsupported file type: {ext}")

        except Exception as e:
            raise ValueError(f"Error processing file '{file_name}': {str(e)}")

    def _process_csv_file(self, file_path: str, header_row: int) -> pl.DataFrame:
        """Process CSV file with proper encoding and time column handling."""
        try:
            df = pl.read_csv(
                file_path,
                skip_rows=header_row,
                infer_schema_length=None,
                encoding='utf-8-sig'
            )
        except Exception:
            df = pl.read_csv(
                file_path,
                skip_rows=header_row,
                infer_schema_length=None,
                encoding='utf-8'
            )

        if not df.is_empty() and 'Time' in df.columns:
            df = df.with_columns([
                pl.col('Time')
                .str.strptime(pl.Datetime, format='%d-%m-%y %H:%M', strict=False)
                .alias('Time')
            ])

        return df

    def combine_all(self):
        try:
            if not self.selected_files:
                QMessageBox.warning(self, "Warning", "No files to combine")
                self.update_status("No files to combine")
                return

            # Use changed-header DataFrames if available for each file
            use_changed_headers = (
                hasattr(self, 'processed_dfs_with_changed_headers') and
                isinstance(self.processed_dfs_with_changed_headers, dict) and
                len(self.processed_dfs_with_changed_headers) > 0
            )

            # Get schema from first file (prefer changed header if available)
            first_file = self.selected_files[0]
            if use_changed_headers and first_file in self.processed_dfs_with_changed_headers:
                first_df = self.processed_dfs_with_changed_headers[first_file]
            else:
                ext = os.path.splitext(first_file)[1].lower()
                header_row = find_header_start_row(first_file, ext)
                try:
                    first_df = self.process_file(first_file)
                    if first_df.is_empty():
                        raise ValueError("First file is empty or could not be processed")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error processing first file: {str(e)}")
                    self.update_status(f"Error processing first file: {str(e)}")
                    return

            standard_schema = first_df.schema

            processed_dfs = []
            error_files = []
            for file in self.selected_files:
                try:
                    if use_changed_headers and file in self.processed_dfs_with_changed_headers:
                        df = self.processed_dfs_with_changed_headers[file]
                    else:
                        df = self.process_file(file)
                    if not df.is_empty():
                        aligned_df = align_dataframe_to_schema(df, standard_schema)
                        processed_dfs.append(aligned_df)
                    else:
                        error_files.append(f"{os.path.basename(file)} (empty file)")
                except Exception as e:
                    error_files.append(f"{os.path.basename(file)} ({str(e)})")
                    continue

            if not processed_dfs:
                QMessageBox.critical(self, "Error", "No valid dataframes to combine")
                self.update_status("No valid dataframes to combine")
                return

            try:
                combined_df = pl.concat(processed_dfs)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error combining dataframes: {str(e)}")
                self.update_status(f"Error combining dataframes: {str(e)}")
                return

            self.combined_df = combined_df
            if COMBINED_TEMP_KEY not in self.selected_files:
                self.selected_files.insert(0, COMBINED_TEMP_KEY)
            self.update_listbox()
            self.update_button_states()
            self.update_status(f"Successfully combined {len(processed_dfs)} files. Preview updated.")
            success_msg = f"Successfully combined {len(processed_dfs)} files."
            if error_files:
                success_msg += "\n\nFiles with errors:\n" + "\n".join(f"- {f}" for f in error_files)
            QMessageBox.information(self, "Success", success_msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error combining files: {str(e)}")
            self.update_status(f"Error combining files: {str(e)}")

    def select_template(self):
        """Select a template file for header mapping."""
        try:
            # Get the template file
            template_file, _ = QFileDialog.getOpenFileName(
                self, "Select Header Mapping Template", "",
                "Excel files (*.xlsx *.xls);;All files (*.*)"
            )
            
            if not template_file:
                self.update_status("Template selection cancelled")
                return
                
            template_file_name = os.path.basename(template_file)
            self.update_status(f"Reading mapping from: {template_file_name}...")
            
            # Read the mapping file using specified engine with fallback
            mapping_df = None
            try:
                mapping_df = pl.read_excel(template_file, engine="openpyxl")
            except ImportError:
                 QMessageBox.critical(
                    self, "Error",
                    "Exporting or loading template from Excel requires 'openpyxl'.\n"
                    "Please install it using:\n\n pip install openpyxl"
                )
                 self.update_status("Template loading failed: openpyxl missing")
                 return
            except Exception as e_openpyxl:
                QMessageBox.critical(
                    self, "Error", 
                    f"Error reading template file with openpyxl: {str(e_openpyxl)}"
                )
                self.update_status("Template loading failed")
                return
            
            if mapping_df is None:
                 QMessageBox.critical(
                     self, "Error",
                     f"Failed to read template file '{template_file_name}'.\n"
                     "Could not read with openpyxl."
                 )
                 self.update_status("Template loading failed")
                 return

            # Check if the mapping file has at least 2 columns
            if mapping_df.width < 2:
                QMessageBox.critical(
                    self, "Error",
                    f"The template file must have at least 2 columns (found {mapping_df.width})!\\n"
                    "Column A should contain original header names.\\n"
                    "Column B should contain new header names."
                )
                self.update_status("Template loading failed: invalid format")
                return
                
            # Extract mapping from columns A and B
            # Get column names (they might not be named 'A' and 'B')
            # Use schema from the read dataframe
            col_names = mapping_df.columns
            if len(col_names) < 2:
                QMessageBox.critical(
                    self, "Error", 
                    "Template file doesn't have enough columns!"
                )
                return
                
            # Create mapping dictionary from first two columns
            mapping = {}
            filename_mapping = {}  # New dictionary to store filename-specific mappings
            
            # Store columns to delete from column 4 if it exists
            self.columns_to_delete = []
            column_names = mapping_df.columns
            if mapping_df.width >= 4: # Changed from 3 to 4, expecting 4 columns now
                # delete_col = mapping_df.get_column(3)  # Get 4th column (0-based index)
                delete_col = mapping_df[column_names[3]]
                self.columns_to_delete = [str(col).strip() for col in delete_col if str(col).strip()]
            
            for row in mapping_df.rows():
                original = str(row[0]).strip() if row[0] is not None else ""
                new_name = str(row[1]).strip() if row[1] is not None else ""
                filename = str(row[2]).strip() if len(row) > 2 and row[2] is not None else ""
                
                if original and new_name:  # Only add if both values exist
                    if filename:  # If there's a filename specified
                        if filename not in filename_mapping:
                            filename_mapping[filename] = {}
                        filename_mapping[filename][original] = new_name
                    else:  # If no filename specified, add to general mapping
                        mapping[original] = new_name
            
            if not mapping and not filename_mapping:
                QMessageBox.critical(
                    self, "Error",
                    "No valid header mappings found in the template file!"
                )
                self.update_status("Template loading failed: no valid mappings")
                return
                
            # Store template file and mappings
            self.template_file = template_file
            self.header_mapping = mapping
            self.filename_mapping = filename_mapping  # Store filename-specific mappings
            
            # Show success message with mapping count
            total_mappings = len(mapping) + sum(len(m) for m in filename_mapping.values())
            filename_count = len(filename_mapping)
            
            success_msg = f"Template file '{template_file_name}' loaded successfully!\n\n"
            success_msg += f"Found {total_mappings} total header mappings.\n"
            if filename_count > 0:
                success_msg += f"Of which {filename_count} are filename-specific mappings."
            
            if self.columns_to_delete:
                success_msg += f"\n\nFound {len(self.columns_to_delete)} columns marked for deletion."
            
            QMessageBox.information(self, "Template Loaded", success_msg)
            
            self.update_status(f"Template loaded: {total_mappings} mappings")
            
            # Update button states
            self.update_button_states()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Error loading template file: {str(e)}"
            )
            self.update_status("Template loading failed")

    def delete_columns(self):
        """Delete columns based on the list from template file."""
        try:
            # Check if we have files to process
            if not self.selected_files:
                QMessageBox.warning(self, "Warning", "Please select files first!")
                return

            # Check if we have combined dataframe for multiple files
            if len(self.selected_files) > 1 and self.combined_df is None:
                QMessageBox.warning(self, "Warning", "Please combine files first before deleting columns!")
                return

            # Check if we have columns to delete from template
            if not hasattr(self, 'columns_to_delete') or not self.columns_to_delete:
                QMessageBox.warning(
                    self, "Warning",
                    "No columns marked for deletion in the template file!\\n"
                    "Please select a template file with column 4 containing column names to delete."
                )
                return

            # Process single file
            if len(self.selected_files) == 1:
                df = self.process_file(self.selected_files[0])
                if df.is_empty():
                    QMessageBox.critical(self, "Error", "Failed to process the selected file!")
                    return

                # Delete only existing columns
                existing_columns = [col for col in self.columns_to_delete if col in df.columns]
                if not existing_columns: # Ensure the list is not empty after filtering
                    QMessageBox.information(self, "Info", "None of the specified columns were found in the file!")
                    return

                df = df.drop(existing_columns)
                self.df = df  # Store back for further use
                QMessageBox.information(self, "Success", f"Deleted {len(existing_columns)} columns from the file.")

            else:
                # Multiple files case
                existing_columns = [col for col in self.columns_to_delete if col in self.combined_df.columns]
                if not existing_columns:
                    QMessageBox.information(self, "Info", "None of the specified columns were found in the combined data!")
                    return
  
                self.combined_df = self.combined_df.drop(existing_columns)
                QMessageBox.information(self, "Success", f"Deleted {len(existing_columns)} columns from combined data.")
                
                # Update data preview after combined file column deletion
                self.original_file_df_for_preview = self.combined_df # For combined data, this is the new 'original' view
                self.changed_header_df_for_preview = None
                self.update_button_states() # Update button states, including switch state
  
            self.update_status(f"Deleted {len(existing_columns)} columns.")
  
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during column deletion: {str(e)}")
            self.update_status("Column deletion failed")
            self.update_button_states() # Update button states on error

    def on_file_selected_for_display(self, item):
        file_key = item.data(Qt.ItemDataRole.UserRole)
        if file_key == COMBINED_TEMP_KEY:
            if self.combined_df is not None:
                self.current_preview_df = self.combined_df
                self.last_displayed_file_path = COMBINED_TEMP_KEY
                self.display_data_preview(self.combined_df)
                self.update_status("Displaying combined data")
            else:
                self.display_data_preview(pl.DataFrame())
                self.update_status("No combined data available")
            return
        # Ensure that 'item' is a QListWidgetItem
        if not isinstance(item, QListWidgetItem):
            self.update_status("Invalid item selected for display.")
            return

        full_file_path = file_key
        if not full_file_path or full_file_path not in self.selected_files:
            self.update_status(f"Could not find full path for: {file_key}")
            return

        # Prevent reloading if the same file is already displayed
        if self.last_displayed_file_path == full_file_path:
            return

        # --- Caching ---
        if full_file_path in self.dataframe_cache:
            self.update_status(f"Displaying cached data for: {os.path.basename(full_file_path)}")
            cached_df = self.dataframe_cache[full_file_path]
            self._update_preview_ui_after_load(cached_df, full_file_path)
            return

        # --- Debouncing ---
        self.pending_load_path = full_file_path
        self.load_delay_timer.start(250)  # Wait 250ms before loading
        self.show_loading()

    def _start_loading_file(self):
        if not self.pending_load_path:
            return

        file_path = self.pending_load_path
        file_name_only = os.path.basename(file_path)
        self.update_status(f"Loading data for: {file_name_only}...")

        # Start a new thread for data processing
        if self.data_worker and self.data_worker.isRunning():
            self.data_worker.quit()
            self.data_worker.wait()

        self.data_worker = DataLoadWorker(file_path, self.process_file)
        self.data_worker.data_loaded_signal.connect(self._update_preview_ui_after_load)
        self.data_worker.error_signal.connect(lambda msg: QMessageBox.critical(self, "Processing Error", msg))
        self.data_worker.start()

    def _update_preview_ui_after_load(self, original_df: pl.DataFrame, file_path: str):
        # Cache the newly loaded dataframe
        if file_path not in self.dataframe_cache:
            self.dataframe_cache[file_path] = original_df

        self.last_displayed_file_path = file_path
        self.hide_loading()

        try:
            # When a single file is loaded, we always want to prepare its data
            self.original_file_df_for_preview = original_df
            self.display_data_preview(self.original_file_df_for_preview)
        except Exception as e:
            QMessageBox.critical(
                self, "Display Error", 
                f"Error updating data preview UI: {str(e)}"
            )
            self.update_status("Data preview update failed")

    def display_data_preview(self, df: pl.DataFrame):
        """Display data preview in the QTableWidget."""
        # Ensure self.data_table exists before proceeding
        if not hasattr(self, 'data_table') or self.data_table is None:
            QMessageBox.critical(self, "Table Initialization Error", "Data table widget is not initialized.")
            return

        self.data_table.clearContents()
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)
        # self.update()  # Removed to reduce flicker

        if df.is_empty():
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(1)
            self.data_table.setHorizontalHeaderLabels(["No Data to Display"])
            return

        columns = [str(col) for col in df.columns]
        self.data_table.setColumnCount(len(columns))
        self.data_table.setHorizontalHeaderLabels(columns)

        # Populate data (only first 100 rows for preview performance)
        self.data_table.setRowCount(min(df.height, 100))
        for r_idx, row in enumerate(df.head(100).iter_rows()):
            for c_idx, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.data_table.setItem(r_idx, c_idx, item)

        # Adjust column widths to content
        self.data_table.resizeColumnsToContents()

    def show_loading(self):
        # Show only a centered 'Loading Data...' message in the data_table
        if getattr(self, 'data_table', None) is not None:
            self.data_table.clearContents()
            self.data_table.setRowCount(1)
            self.data_table.setColumnCount(1)
            self.data_table.setHorizontalHeaderLabels(["Loading Data..."])
            loading_item = QTableWidgetItem("Loading Data...")
            loading_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.data_table.setItem(0, 0, loading_item)
            self.data_table.setVisible(True)

    def hide_loading(self):
        # No-op: the table will be updated by the data preview logic after loading
        pass

def find_header_start_row(file_path: str, ext: str, num_rows_to_check: int = 100) -> int:
    """
    Find the header row by:
    1. Reading first 100 rows or entire file
    2. Remove rows where first column is empty
    3. Find first row with maximum number of columns
    """
    try:
        # Read sample data
        if ext == '.csv':
            # Read entire file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
            
            # Process rows
            valid_rows = []  # Will store tuples of (row_index, number_of_columns)
            
            # Process only first num_rows_to_check if file is larger
            lines_to_check = all_lines[:num_rows_to_check]
            
            for i, line in enumerate(lines_to_check):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Split and clean columns
                columns = [col.strip() for col in line.split(',')]
                
                # Skip if first column is empty
                if not columns[0]:
                    continue
                
                # Count non-empty columns
                non_empty_count = len([col for col in columns if col])
                if non_empty_count > 0:
                    valid_rows.append((i, non_empty_count))
            
            if not valid_rows:
                return 0
            
            # Find the row with maximum columns
            max_cols = max(count for _, count in valid_rows)
            
            # Return the first row that has the maximum number of columns
            for row_idx, count in valid_rows:
                if count == max_cols:
                    return row_idx
            
            return 0

        elif ext in ['.xlsx', '.xls']:
            # Read entire file
            df = pl.read_excel(file_path)
            
            # Process only first num_rows_to_check
            rows_to_check = min(num_rows_to_check, df.height)
            valid_rows = []
            
            for i in range(rows_to_check):
                row = df.row(i)
                
                # Skip if first column is empty
                if row[0] is None or str(row[0]).strip() == "":
                    continue
                
                # Count non-empty cells
                non_empty_count = sum(1 for val in row if val is not None and str(val).strip() != "")
                if non_empty_count > 0:
                    valid_rows.append((i, non_empty_count))
            
            if not valid_rows:
                return 0
            
            # Find row with maximum columns
            max_cols = max(count for _, count in valid_rows)
            
            # Return first row with maximum columns
            for row_idx, count in valid_rows:
                if count == max_cols:
                    return row_idx
            
            return 0

    except Exception as e:
        # Consolidated error handling for any issues during header detection
        return 0

def align_dataframe_to_schema(df: pl.DataFrame, standard_schema: Any) -> pl.DataFrame:
    """
    Aligns a DataFrame's columns to a standard schema using Polars built-in functions.
    """
    try:
        # Create expressions for missing columns
        missing_cols = [
            pl.lit(None, dtype=standard_schema[col]).alias(col)
            for col in standard_schema.keys()
            if col not in df.columns
        ]

        # Add missing columns and select in correct order
        df = (df
            .with_columns(missing_cols if missing_cols else [])\
            .select(list(standard_schema.keys()))
        )

        # Create cast expressions for type mismatches
        cast_exprs = [
            pl.col(col).cast(dtype, strict=False)
            for col, dtype in standard_schema.items()
            if df.schema[col] != dtype
        ]

        # Apply casts if needed
        if cast_exprs:
            df = df.with_columns(cast_exprs)

        # Remove double quotes from string columns after alignment
        string_columns = [col for col in df.columns if df.schema[col] == pl.String]
        if string_columns:
            df = df.with_columns([
                pl.col(col).str.replace('"', '') for col in string_columns
            ])

        return df

    except Exception as e:
        # Attempt to get schema names for error message, handling potential AttributeError
        schema_names_str = "Unknown Schema" # Default
        try:
            # Check if standard_schema has a .names() method (Polars Schema)
            if hasattr(standard_schema, 'names'):
                schema_names_str = str(standard_schema.names())
            # If not, try treating it as a dictionary and getting keys (OrderedDict)
            elif isinstance(standard_schema, dict):
                schema_names_str = str(list(standard_schema.keys()))
            # Otherwise, just show the type
            else:
                schema_names_str = f"Type: {type(standard_schema).__name__}"
        except Exception as schema_info_e:
            print(f"Warning: Could not get schema info for error message: {schema_info_e}") # Keep print for debug
            schema_names_str = "Error retrieving schema info"

        raise ValueError(f"Error aligning DataFrame to schema (Standard Schema: {schema_names_str}): {e}")

def main():
    app = QApplication(sys.argv)
    window = DataProcessingApp()

    try:
        # Set window icon
        if getattr(sys, 'frozen', False):
            # If running as exe
            base_path = sys._MEIPASS
        else:
            # If running as script
            base_path = os.path.dirname(os.path.abspath(__file__))
            
        icon_path = os.path.join(base_path, "icon.ico")
        if os.path.exists(icon_path):
            window.setWindowIcon(QIcon(icon_path))
    except Exception as e:
        print(f"Warning: Could not load icon: {e}")
    
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()