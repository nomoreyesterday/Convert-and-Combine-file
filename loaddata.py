import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import polars as pl
import os
from typing import List, Dict, Any, Optional, Tuple
import sys
# คำสั่ง run exe
# pyinstaller -F loaddata.py -w --add-data "icon.ico;." -i "icon.ico" -n "Data Processing"
# Set appearance mode and color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

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


# This is a heuristic and might need adjustment based on actual file formats
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
            .with_columns(missing_cols if missing_cols else [])
            .select(list(standard_schema.keys())))

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
            # print(f"Warning: Could not get schema info for error message: {schema_info_e}")
            schema_names_str = "Error retrieving schema info"

        raise ValueError(f"Error aligning DataFrame to schema (Standard Schema: {schema_names_str}): {e}")


class DataProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Processing")
        self.root.geometry("800x500")
        
        # Data storage
        self.selected_files: List[str] = []
        self.combined_df: Optional[pl.DataFrame] = None
        self.remove_duplicates_var = tk.BooleanVar(value=True)
        
        # Template mapping storage
        self.template_file: Optional[str] = None
        self.header_mapping: Dict[str, str] = {}

        # Configure theme colors based on appearance mode
        self._configure_theme_colors()
        
        # Configure root and create main layout
        self.root.configure(bg=self.bg_color)
        self._create_main_layout()
        
        # Initialize button states
        self.update_button_states()

    def _configure_theme_colors(self):
        """Configure theme colors based on appearance mode."""
        if ctk.get_appearance_mode() == "Dark":
            self.bg_color = COLORS["primary"]
            self.surface_color = COLORS["primary_dark"]
            self.text_color = COLORS["text_light"]
            self.sidebar_color = COLORS["primary_dark"]
        else:
            self.bg_color = COLORS["background"]
            self.surface_color = COLORS["surface"]
            self.text_color = COLORS["text"]
            self.sidebar_color = COLORS["sidebar"]

    def _create_main_layout(self):
        """Create the main application layout."""
        # Create main container
        self.main_container = ctk.CTkFrame(
            self.root,
            fg_color=self.surface_color,
            corner_radius=0
        )
        self.main_container.pack(fill="both", expand=True)

        # Create sidebar and main content
        self.create_sidebar()
        self.create_main_content()

    def create_sidebar(self):
        # Sidebar container
        sidebar = ctk.CTkFrame(
            self.main_container,
            fg_color=self.sidebar_color,
            width=200,
            corner_radius=0
        )
        sidebar.pack(side="left", fill="y", padx=0, pady=0)
        sidebar.pack_propagate(False)

        # App title
        app_title = ctk.CTkLabel(
            sidebar,
            text="Data Processing",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_light"]
        )
        app_title.pack(pady=10, padx=20)

        # Action buttons
        select_button = self.create_sidebar_button(
            sidebar,
            "Select Files",
            self.select_files,
            COLORS["secondary"],
            hover_color=COLORS["secondary_dark"],
            text_color=COLORS["text_light"],
            border_width=2,
            border_color=COLORS["secondary_dark"]
        )

        combine_button = self.create_sidebar_button(
            sidebar,
            "Combine Files",
            self.combine_all,
            COLORS["secondary"],
            hover_color=COLORS["secondary_dark"],
            text_color=COLORS["text_light"],
            border_width=2,
            border_color=COLORS["secondary_dark"]
        )

        delete_column_button = self.create_sidebar_button(
            sidebar,
            "Delete Columns",
            self.delete_columns,
            COLORS["secondary"],
            hover_color=COLORS["secondary_dark"],
            text_color=COLORS["text_light"],
            border_width=2,
            border_color=COLORS["secondary_dark"]
        )

        clear_button = self.create_sidebar_button(
            sidebar,
            "Clear All",
            self.clear_files,
            COLORS["secondary"],
            hover_color=COLORS["secondary_dark"],
            text_color=COLORS["text_light"],
            border_width=2,
            border_color=COLORS["secondary_dark"]
        )

        # Status section
        status_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        status_frame.pack(side="bottom", fill="x", padx=10, pady=20)

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Status: Ready",
            font=ctk.CTkFont(size=12),
            wraplength=180,
            text_color=COLORS["text_light"]
        )
        self.status_label.pack(pady=5)

    def create_sidebar_button(self, parent, text, command, fg_color, **kwargs):
        btn = ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=160,
            height=40,
            fg_color=fg_color,
            corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),  # Made text bold
            **kwargs
        )
        btn.pack(pady=10, padx=20)
        return btn

    def create_main_content(self):
        """Create the main content area with file list and controls."""
        content = self._create_content_container()
        header_frame = self._create_header_section(content)
        self._create_file_list(content)

    def _create_content_container(self) -> ctk.CTkFrame:
        """Create and return the main content container."""
        content = ctk.CTkFrame(
            self.main_container,
            fg_color=self.surface_color,
            corner_radius=0
        )
        content.pack(side="left", fill="both", expand=True, padx=0, pady=0)
        return content

    def _create_header_section(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Create the header section with title, select template and export buttons."""
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(15, 0))  # ลด pady ด้านล่างเป็น 0

        # Title and subtitle
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side="left", fill="x", expand=True)

        title_label = ctk.CTkLabel(
            title_frame,
            text="Selected Files",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["primary"]
        )
        title_label.pack(anchor="w")

        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Select and combine your data files",
            font=ctk.CTkFont(size=12),
            text_color="gray50"
        )
        subtitle_label.pack(anchor="w")

        # Buttons frame (for multiple buttons)
        buttons_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        buttons_frame.pack(side="right", padx=0, pady=5)
        
        # Select Template button
        self.select_template_btn = ctk.CTkButton(
            buttons_frame,
            text="Select Template",
            command=self.select_template,
            width=120,
            height=32,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_dark"],
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8
        )
        self.select_template_btn.pack(side="left", padx=(0, 5))
        
        # Change Header button
        self.change_header_btn = ctk.CTkButton(
            buttons_frame,
            text="Change Header",
            command=self.change_header,
            width=120,
            height=32,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary"],  # Same as fg_color to disable hover effect initially
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8
        )
        self.change_header_btn.pack(side="left", padx=(0, 5))

        # Export button
        self.export_btn = ctk.CTkButton(
            buttons_frame,
            text="Export File",
            command=self.export_file,
            width=120,
            height=32,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary"],  # Same as fg_color to disable hover effect initially
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8
        )
        self.export_btn.pack(side="right", padx=0)
        
        return header_frame

    def _create_file_list(self, parent: ctk.CTkFrame):
        """Create the file list display area."""
        # Create main container for file list section
        file_list_container = ctk.CTkFrame(parent, fg_color="transparent")
        file_list_container.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        # Add duplicate rows checkbox
        duplicate_check = ctk.CTkCheckBox(
            file_list_container,
            text="Remove duplicate rows when combining files",
            variable=self.remove_duplicates_var,
            text_color=self.text_color,
            fg_color=COLORS["primary"],
            hover_color=COLORS["secondary"],
            border_color=COLORS["primary"],
            checkmark_color=COLORS["text_light"],
            font=ctk.CTkFont(size=12),
            checkbox_width=16,
            checkbox_height=16,
        )
        duplicate_check.pack(anchor="w", padx=5, pady=(2, 5))  # ปรับ pady เป็น (2, 5)

        # Add separator
        separator = ctk.CTkFrame(file_list_container, height=1, fg_color=COLORS["secondary"])
        separator.pack(fill="x", pady=(0, 5))

        # Create listbox
        self.listbox = tk.Listbox(
            file_list_container,
            bg=self.surface_color,
            fg=self.text_color,
            font=("Segoe UI", 11),
            relief="flat",
            borderwidth=0,
            selectmode="extended",
            selectbackground=COLORS["primary"],
            highlightthickness=1,
            highlightcolor=COLORS["primary"],
            activestyle='none'
        )
        self.listbox.pack(fill="both", expand=True)

    def create_section(self, parent, title, subtitle=""):
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=10, pady=10)

        # Title with subtitle
        title_frame = ctk.CTkFrame(section, fg_color="transparent")
        title_frame.pack(fill="x", padx=5, pady=(0, 10))

        title_label = ctk.CTkLabel(
            title_frame,
            text=title,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["primary"]
        )
        title_label.pack(anchor="w")

        if subtitle:
            subtitle_label = ctk.CTkLabel(
                title_frame,
                text=subtitle,
                font=ctk.CTkFont(size=12),
                text_color="gray50"
            )
            subtitle_label.pack(anchor="w")

        return section

    def update_button_states(self):
        """Update the state of action buttons based on whether files are selected and template is loaded."""
        has_files = len(self.selected_files) > 0
        # Check if a template file is loaded AND if there is at least one mapping (either general or filename-specific)
        has_template = self.template_file is not None and (self.header_mapping or (hasattr(self, 'filename_mapping') and self.filename_mapping))
        
        if has_files:
            # Enable export button if files are selected
            self.export_btn.configure(state="normal", fg_color=COLORS["primary"], hover_color=COLORS["primary_dark"])
            
            # Enable select template button when files are selected
            self.select_template_btn.configure(state="normal", fg_color=COLORS["primary"], hover_color=COLORS["primary_dark"])
            
            # Enable change header button only if both files and template are available
            if has_template:
                self.change_header_btn.configure(state="normal", fg_color=COLORS["primary"], hover_color=COLORS["primary_dark"])
            else:
                self.change_header_btn.configure(state="disabled", fg_color="#f2f2f2")
        else:
            # Disable all buttons if no files are selected
            self.select_template_btn.configure(state="disabled", fg_color="#f2f2f2")
            self.change_header_btn.configure(state="disabled", fg_color="#f2f2f2")
            self.export_btn.configure(state="disabled", fg_color="#f2f2f2")

    def select_files(self):
        """Opens file dialog and adds selected files to the listbox."""
        try:
            files = filedialog.askopenfilenames(
                title="Select Files",
                filetypes=[
                    ("All supported files", "*.parquet *.csv *.xlsx *.xls"),
                    ("Parquet files", "*.parquet"),
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx *.xls"),
                    ("All files", "*.*"),
                ]
            )

            if files:
                # Add new files to the list
                for file in files:
                    if file not in self.selected_files:
                        self.selected_files.append(file)
                
                # Update listbox
                self.update_listbox()
                self.update_status(f"Added {len(files)} new files")
                
                # Reset combined_df when new files are added
                self.combined_df = None
                
                # Update button states
                self.update_button_states()

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error during file selection: {str(e)}"
            )
            self.update_status("Error selecting files")

    def update_listbox(self):
        """Updates the listbox with current selected files."""
        if self.listbox:
            self.listbox.delete(0, tk.END)
            for file in self.selected_files:
                # Add padding spaces before the filename
                self.listbox.insert(tk.END, f"    {os.path.basename(file)}")  # Added 4 spaces for padding

    def clear_files(self):
        """Clears all selected files."""
        self.selected_files = []
        if self.listbox:
            self.listbox.delete(0, tk.END)
        self.combined_df = None
        self.update_status("Cleared all files")
        
        # Update button states
        self.update_button_states()

    def export_file(self):
        """Export the processed files with various options."""
        if not self.selected_files:
            messagebox.showwarning(
                "Warning",
                "Please select files first before exporting!"
            )
            return
            
        try:
            # Create export options dialog
            export_dialog = ctk.CTkToplevel(self.root)
            export_dialog.title("Export Options")
            export_dialog.geometry("400x400")
            export_dialog.transient(self.root)
            export_dialog.grab_set()

            # Center the dialog
            window_width = 400
            window_height = 400
            screen_width = export_dialog.winfo_screenwidth()
            screen_height = export_dialog.winfo_screenheight()
            center_x = int(screen_width/2 - window_width/2)
            center_y = int(screen_height/2 - window_height/2)
            export_dialog.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

            # Export options
            export_frame = ctk.CTkFrame(export_dialog, fg_color="transparent")
            export_frame.pack(fill="both", expand=True, padx=20, pady=20)

            # Export mode selection
            mode_label = ctk.CTkLabel(
                export_frame,
                text="Export Mode:",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            mode_label.pack(anchor="w", pady=(0, 5))

            export_mode = tk.StringVar(value="combined")
            mode_frame = ctk.CTkFrame(export_frame, fg_color="transparent")
            mode_frame.pack(fill="x", pady=(0, 15))

            combined_radio = ctk.CTkRadioButton(
                mode_frame,
                text="Export as single combined file",
                variable=export_mode,
                value="combined",
                font=ctk.CTkFont(size=12)
            )
            combined_radio.pack(anchor="w", pady=2)

            separate_radio = ctk.CTkRadioButton(
                mode_frame,
                text="Export as separate files",
                variable=export_mode,
                value="separate",
                font=ctk.CTkFont(size=12)
            )
            separate_radio.pack(anchor="w", pady=2)

            # File naming options
            naming_label = ctk.CTkLabel(
                export_frame,
                text="File Naming:",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            naming_label.pack(anchor="w", pady=(0, 5))

            naming_frame = ctk.CTkFrame(export_frame, fg_color="transparent")
            naming_frame.pack(fill="x", pady=(0, 15))

            naming_mode = tk.StringVar(value="original")
            original_radio = ctk.CTkRadioButton(
                naming_frame,
                text="Keep original filenames",
                variable=naming_mode,
                value="original",
                font=ctk.CTkFont(size=12)
            )
            original_radio.pack(anchor="w", pady=2)

            prefix_radio = ctk.CTkRadioButton(
                naming_frame,
                text="Add prefix to filenames",
                variable=naming_mode,
                value="prefix",
                font=ctk.CTkFont(size=12)
            )
            prefix_radio.pack(anchor="w", pady=2)

            prefix_entry = ctk.CTkEntry(
                naming_frame,
                placeholder_text="Enter prefix (e.g., 'processed_')",
                width=200
            )
            prefix_entry.pack(anchor="w", pady=2)

            # Export button
            def do_export():
                mode = export_mode.get()
                naming = naming_mode.get()
                prefix = prefix_entry.get().strip() if naming == "prefix" else ""

                export_dialog.destroy()

                if mode == "combined":
                    self._export_combined_file()
                else:
                    self._export_separate_files(naming, prefix)

            export_btn = ctk.CTkButton(
                export_frame,
                text="Export",
                command=do_export,
                width=120,
                height=32,
                fg_color=COLORS["primary"],
                hover_color=COLORS["primary_dark"]
            )
            export_btn.pack(pady=20)

        except Exception as e:
            messagebox.showerror(
                "Error", 
                f"Error preparing export: {str(e)}"
            )
            self.update_status("Export Failed")

    def _export_combined_file(self):
        """Export all files as a single combined file."""
        if self.combined_df is None:
            messagebox.showwarning(
                "Warning",
                "Please combine files first using the 'Combine Files' button before exporting a combined file!"
            )
            return

        try:
            save_path = self._get_save_path()
            if not save_path:
                return

            self._save_dataframe(save_path)

        except Exception as e:
            messagebox.showerror(
                "Error", 
                f"Error exporting combined file: {str(e)}"
            )
            self.update_status("Export Failed")

    def _export_separate_files(self, naming_mode: str, prefix: str = ""):
        """Export files separately with specified naming convention."""
        try:
            # Get save directory
            save_dir = filedialog.askdirectory(
                title="Select Directory to Save Files"
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
                messagebox.showinfo("Export Complete", success_msg)
                self.update_status(f"Exported {success_count} files")
            else:
                messagebox.showerror(
                    "Export Failed",
                    "No files were successfully exported.\n\nFailed files:\n" + 
                    "\n".join(f"- {f}" for f in error_files)
                )
                self.update_status("Export Failed")

        except Exception as e:
            messagebox.showerror(
                "Error",
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
            messagebox.showwarning(
                "Warning",
                "Please select a file first before changing headers!"
            )
            return
        
        # Check if a template has been loaded
        if not self.template_file or (not self.header_mapping and not hasattr(self, 'filename_mapping')):
            messagebox.showwarning(
                "Warning",
                "Please select a template file first using the 'Select Template' button!"
            )
            return
            
        try:
            # Process each file
            processed_files = []
            skipped_files = []
            total_changes = 0
            self.processed_dfs_with_changed_headers: List[pl.DataFrame] = [] # Store processed dataframes
            
            for source_file in self.selected_files:
                source_file_name = os.path.basename(source_file)
                
                # Get the appropriate mapping for this file
                # Find a filename-specific mapping that is a substring of the source file name
                mapping = self.header_mapping # Start with general mapping
                applied_mapping_name = "general mapping"
                
                if hasattr(self, 'filename_mapping'):
                    found_specific_mapping = False
                    # Check for exact match first (though less likely with your filenames)
                    if source_file_name in self.filename_mapping:
                        mapping = self.filename_mapping[source_file_name]
                        applied_mapping_name = f"specific mapping for '{source_file_name}'"
                        found_specific_mapping = True
                    else:
                        # Check if any filename in the mapping is a substring of the source file name
                        for template_filename, specific_mapping in self.filename_mapping.items():
                            if template_filename in source_file_name:
                                mapping = specific_mapping
                                applied_mapping_name = f"specific mapping for filename containing '{template_filename}'"
                                found_specific_mapping = True
                                break # Use the first match found

                if not mapping:
                    skipped_files.append(source_file_name)
                    continue
                
                # Process the file
                try:
                    df = self.process_file(source_file)
                    if df.is_empty():
                        messagebox.showerror(
                            "Error",
                            f"The source file '{source_file_name}' resulted in an empty DataFrame!"
                        )
                        continue
                except Exception as e:
                    messagebox.showerror(
                        "Processing Error",
                        f"Error processing file '{source_file_name}':\n\n{str(e)}"
                    )
                    self.update_status(f"Processing failed: {source_file_name}")
                    continue
                
                # Apply the mapping to rename columns
                old_columns = df.columns
                new_columns = []
                seen_names = {}
                
                for old_col in old_columns:
                    # Get the new column name from mapping or keep the original if not found
                    new_col = mapping.get(old_col, old_col)
                    
                    # Handle duplicate target column names
                    if new_col in seen_names:
                        seen_names[new_col] += 1
                        new_col = f"{new_col}_{seen_names[new_col]}"
                    else:
                        seen_names[new_col] = 0
                    
                    new_columns.append(new_col)
                
                # Create rename mapping
                rename_mapping = dict(zip(old_columns, new_columns))
                renamed_df = df.rename(rename_mapping)
                
                # Store the dataframe with changed headers
                self.processed_dfs_with_changed_headers.append(renamed_df)
                
                # Count changes
                changes = sum(1 for old, new in zip(old_columns, new_columns) if old != new)
                total_changes += changes
                
                if changes > 0:
                    processed_files.append((source_file_name, changes))
                
            # Show success message with statistics
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
            
            messagebox.showinfo("Success", "\n".join(success_msg))
            self.update_status(f"Changed headers in {len(processed_files)} files")
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error changing headers: {str(e)}"
            )
            self.update_status("Header change failed")

    def _get_save_path(self) -> Optional[str]:
        """Get the save path from file dialog."""
        return filedialog.asksaveasfilename(
            initialfile="combined_data",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*"),
            ]
        )

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
                messagebox.showerror(
                    "Export Error", 
                    "Exporting to Excel requires 'openpyxl'. Please install it:\n\npip install openpyxl"
                )
                self.update_status("Export Failed (Excel: openpyxl missing)")
                return
        else:
            messagebox.showwarning(
                "Export Warning", 
                f"Unsupported file type: {ext}\nPlease select a supported type (parquet, csv, or xlsx)."
            )
            self.update_status("Export Failed (Unsupported type)")
            return

        messagebox.showinfo(
            "Success", 
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

            self.status_label.configure(
                text=f"Status: {message}",
                text_color=color
            )
            self.root.update_idletasks()

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
                try:
                    # Find header row
                    header_row = find_header_start_row(file_path, ext)
                    
                    if ext == '.csv':
                        df = self._process_csv_file(file_path, header_row)
                    else:
                        try:
                            # For Excel files, use fastexcel
                            df = pl.read_excel(
                                file_path, 
                                offset=header_row,
                                read_options={"has_header": True}
                            )
                        except Exception as excel_err:
                            # print(f"Error reading Excel file with fastexcel: {excel_err}")
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

                except Exception as inner_err:
                    # print(f"Inner processing error: {inner_err}")
                    # Last resort fallback
                    if ext == '.csv':
                        df = pl.read_csv(file_path)
                    else:
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
            # print(f"Error processing file: {e}")
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
        """
        Combines all selected files into a single DataFrame.
        Processes each file, determines a standard schema from the first file,
        aligns all DataFrames to that schema, and concatenates them.
        Also checks and removes duplicate rows if the option is selected.
        If headers have been changed, uses the processed dataframes with changed headers.
        """
        if not self.selected_files:
            messagebox.showwarning("Warning", "No files selected to combine!")
            self.combined_df = None
            self.update_status("Combine All: No files selected")
            return

        self.update_status(f"Starting combine process for {len(self.selected_files)} files...")
        
        processed_dfs: List[pl.DataFrame] = []
        standard_schema: Optional[pl.Schema] = None
        processing_errors: List[str] = []
        alignment_errors: List[str] = []
        successful_file_names: List[str] = []
        
        # Determine which dataframes to combine
        if hasattr(self, 'processed_dfs_with_changed_headers') and self.processed_dfs_with_changed_headers:
            processed_dfs = self.processed_dfs_with_changed_headers
            self.update_status("Combining dataframes with changed headers...")
            # Need to determine standard schema from these processed dataframes
            standard_df = max(processed_dfs, key=lambda df: df.width)
            standard_schema = standard_df.schema
            print(f"Standard schema set from processed dataframes: {standard_df.columns}")
        else:
            # If headers haven't been changed, process files from scratch
            # --- Step 1: Process all files ---
            print("\n--- Processing Individual Files ---")
            for i, file_path in enumerate(self.selected_files):
                file_name = os.path.basename(file_path)
                try:
                    self.update_status(f"Processing file {i+1}/{len(self.selected_files)}: {file_name}...")
                    df = self.process_file(file_path)

                    # Check if dataframe is empty after processing
                    if df.is_empty():
                        print(f"File '{file_name}' processed but resulted in an empty DataFrame. Skipping.")
                        continue

                    processed_dfs.append(df)
                    successful_file_names.append(file_name)

                    # Set standard schema from the first successfully processed non-empty DataFrame
                    if standard_schema is None:
                        standard_schema = df.schema
                        print(f"Standard schema set from '{file_name}': {standard_schema.names()}")

                except Exception as e:
                    error_msg = f"Failed to process file '{file_name}': {e}"
                    print(error_msg)
                    processing_errors.append(error_msg)

            print("--- Finished Processing Individual Files ---")

        if not processed_dfs:
            msg = "No valid dataframes were processed from selected files."
            # Include processing and alignment errors if files were processed from scratch
            if hasattr(self, 'processed_dfs_with_changed_headers') and self.processed_dfs_with_changed_headers:
                final_errors = [] # No processing errors if using pre-processed data
            else:
                final_errors = processing_errors + alignment_errors
            
            messagebox.showerror("Combine Error", msg + "\n\nDetails:\n" + "\n".join(final_errors))
            self.combined_df = None
            self.update_status("Combine All: Failed (No valid data)")
            return

        if standard_schema is None:
            msg = "Internal Error: Could not determine a standard schema."
            if hasattr(self, 'processed_dfs_with_changed_headers') and self.processed_dfs_with_changed_headers:
                final_errors = [] # No processing errors if using pre-processed data
            else:
                final_errors = processing_errors + alignment_errors
            
            messagebox.showerror("Combine Error", msg + "\n\nDetails:\n" + "\n".join(final_errors))
            self.combined_df = None
            self.update_status("Combine All: Failed (Internal schema error)")
            return

        # Safely get and print schema names
        schema_names = []
        try:
            if hasattr(standard_schema, 'names'):
                schema_names = standard_schema.names()
            elif isinstance(standard_schema, dict):
                schema_names = list(standard_schema.keys())
            print(f"\nStandard Schema ({len(schema_names)} columns): {schema_names}")
        except Exception as e:
            print(f"\nCould not determine standard schema names: {e}")

        # --- Step 2: Align all processed DataFrames to the standard schema ---
        self.update_status(f"Aligning {len(processed_dfs)} dataframes to standard schema...")
        print("--- Aligning DataFrames ---")
        aligned_dfs: List[pl.DataFrame] = []
        current_alignment_errors: List[str] = []

        for i, df in enumerate(processed_dfs):
            try:
                aligned_df = align_dataframe_to_schema(df, standard_schema)
                aligned_dfs.append(aligned_df)
            except Exception as e:
                # We don't have original filenames readily available here if using processed_dfs_with_changed_headers
                # A more robust solution would involve storing filenames with the processed dataframes.
                align_error_msg = f"Failed to align dataframe {i+1}: {e}"
                print(align_error_msg)
                current_alignment_errors.append(align_error_msg)

        print("--- Finished Aligning DataFrames ---")

        if not aligned_dfs:
            msg = "No dataframes could be aligned to the standard schema."\
                   "Please ensure your template and selected files are compatible."
            # Display a concise error message to the user
            messagebox.showerror("Combine Error", msg)
            self.combined_df = None
            self.update_status("Combine All: Failed (Alignment issues)")
            return

        # --- Step 3: Vertically concatenate aligned DataFrames and check duplicates ---
        self.update_status("Concatenating dataframes...")
        print("--- Concatenating DataFrames ---")
        try:
            # Concatenate all dataframes
            self.combined_df = pl.concat(aligned_dfs, how="vertical_relaxed")
            total_rows_before = self.combined_df.height
            num_duplicates = 0
            
            # Check and remove duplicates if option is selected
            if self.remove_duplicates_var.get():
                self.update_status("Checking for duplicates...")
                duplicates_mask = self.combined_df.is_duplicated()
                num_duplicates = duplicates_mask.sum()
                
                if num_duplicates > 0:
                    self.combined_df = self.combined_df.unique(maintain_order=True)
            
            # Prepare success message
            success_msg = [
                f"Successfully combined {len(aligned_dfs)} files!",
                f"Initial rows: {total_rows_before}"
            ]
            
            if self.remove_duplicates_var.get():
                success_msg.extend([
                    f"Duplicate rows found: {num_duplicates}",
                    f"Final rows after removing duplicates: {self.combined_df.height}"
                ])
            else:
                success_msg.append(f"Total rows (duplicates kept): {self.combined_df.height}")
                
            success_msg.append(f"Total columns: {self.combined_df.width}")
            
            # Add any processing warnings
            if processing_errors or alignment_errors:
                success_msg.append("\nWarnings during processing:")
                if processing_errors:
                    success_msg.append("Some files failed to process:")
                    success_msg.extend(f"  • {err}" for err in processing_errors)
                if alignment_errors:
                    success_msg.append("Some files had alignment issues:")
                    success_msg.extend(f"  • {err}" for err in alignment_errors)
            
            # Show single message with all information
            messagebox.showinfo(
                "Combine Complete",
                "\n".join(success_msg)
            )
            
            # Update status
            if self.remove_duplicates_var.get():
                self.update_status(f"Combined: {self.combined_df.height} rows (duplicates removed), {self.combined_df.width} columns")
            else:
                self.update_status(f"Combined: {self.combined_df.height} rows (duplicates kept), {self.combined_df.width} columns")

        except Exception as e:
            final_errors = processing_errors + alignment_errors + [f"Concatenation failed: {e}"]
            messagebox.showerror(
                "Combine Error", 
                f"Failed to concatenate dataframes: {e}\n\nDetails:\n" + "\n".join(final_errors)
            )
            self.combined_df = None
            self.update_status("Combine All: Failed (Concatenation)")

    def select_template(self):
        """Select a template file for header mapping."""
        try:
            # Get the template file
            template_file = filedialog.askopenfilename(
                title="Select Header Mapping Template",
                filetypes=[
                    ("Excel files", "*.xlsx *.xls"),
                    ("All files", "*.*")
                ]
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
                 messagebox.showerror(
                    "Error",
                    "Exporting or loading template from Excel requires 'openpyxl'.\n"\
                    "Please install it using:\n\n pip install openpyxl"
                )
                 self.update_status("Template loading failed: openpyxl missing")
                 return
            except Exception as e_openpyxl:
                messagebox.showerror(
                    "Error", 
                    f"Error reading template file with openpyxl: {str(e_openpyxl)}"
                )
                self.update_status("Template loading failed")
                return
            
            if mapping_df is None:
                 messagebox.showerror(
                     "Error",
                     f"Failed to read template file '{template_file_name}'.\n"
                     "Could not read with openpyxl."
                 )
                 self.update_status("Template loading failed")
                 return

            print(f"DEBUG: mapping_df columns after reading: {mapping_df.columns}")
            print(f"DEBUG: mapping_df schema after reading: {mapping_df.schema}")
            # Check if the mapping file has at least 2 columns
            if mapping_df.width < 2:
                messagebox.showerror(
                    "Error",
                    f"The template file must have at least 2 columns (found {mapping_df.width})!\n"
                    "Column A should contain original header names.\n"
                    "Column B should contain new header names."
                )
                self.update_status("Template loading failed: invalid format")
                return
                
            # Extract mapping from columns A and B
            # Get column names (they might not be named 'A' and 'B')
            # Use schema from the read dataframe
            col_names = mapping_df.columns
            if len(col_names) < 2:
                messagebox.showerror(
                    "Error", 
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
                messagebox.showerror(
                    "Error",
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
            
            messagebox.showinfo("Template Loaded", success_msg)
            
            self.update_status(f"Template loaded: {total_mappings} mappings")
            
            # Update button states
            self.update_button_states()
            
        except Exception as e:
            messagebox.showerror(
                "Error", 
                f"Error loading template file: {str(e)}"
            )
            self.update_status("Template loading failed")

    def delete_columns(self):
        """Delete columns based on the list from template file."""
        try:
            # Check if we have files to process
            if not self.selected_files:
                messagebox.showwarning("Warning", "Please select files first!")
                return

            # Check if we have combined dataframe for multiple files
            if len(self.selected_files) > 1 and self.combined_df is None:
                messagebox.showwarning("Warning", "Please combine files first before deleting columns!")
                return

            # Check if we have columns to delete from template
            if not hasattr(self, 'columns_to_delete') or not self.columns_to_delete:
                messagebox.showwarning(
                    "Warning",
                    "No columns marked for deletion in the template file!\n"
                    "Please select a template file with column 4 containing column names to delete."
                )
                return

            # Process single file
            if len(self.selected_files) == 1:
                df = self.process_file(self.selected_files[0])
                if df.is_empty():
                    messagebox.showerror("Error", "Failed to process the selected file!")
                    return

                # Delete only existing columns
                existing_columns = [col for col in self.columns_to_delete if col in df.columns]
                if not existing_columns:
                    messagebox.showinfo("Info", "None of the specified columns were found in the file!")
                    return

                df = df.drop(existing_columns)
                self.df = df  # Store back for further use
                messagebox.showinfo("Success", f"Deleted {len(existing_columns)} columns from the file.")

            else:
                # Multiple files case
                existing_columns = [col for col in self.columns_to_delete if col in self.combined_df.columns]
                if not existing_columns:
                    messagebox.showinfo("Info", "None of the specified columns were found in the combined data!")
                    return

                self.combined_df = self.combined_df.drop(existing_columns)
                messagebox.showinfo("Success", f"Deleted {len(existing_columns)} columns from combined data.")

            self.update_status(f"Deleted {len(existing_columns)} columns.")

        except Exception as e:
            messagebox.showerror("Error", f"Error during column deletion: {str(e)}")
            self.update_status("Column deletion failed")



def main():
    root = ctk.CTk()
    
    # Center the window on screen with adjusted height
    # root.iconbitmap("icon.ico")
    # เปลี่ยนไปมาได้
    try:
        # Get the base path for resources
        if getattr(sys, 'frozen', False):
            # If running as exe
            base_path = sys._MEIPASS
        else:
            # If running as script
            base_path = os.path.dirname(os.path.abspath(__file__))
            
        icon_path = os.path.join(base_path, "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception as e:
        print(f"Warning: Could not load icon: {e}")
    # ถึงตรงนี้
    
    window_width = 800
    window_height = 500
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    app = DataProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()