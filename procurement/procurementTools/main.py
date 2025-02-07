import sys
import os
import pandas as pd
import requests  # Import requests for making API calls
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QMessageBox, QTextEdit, QFrame

class StreetViewDownloaderApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.GSVURL = ""
        self.setWindowTitle("Google Street View Downloader")
        self.setGeometry(100, 100, 1300, 600)  # Increased window width to accommodate the new pane

        # Initialize CSV data and current selection
        self.csv_data = None
        self.current_selection = None
        self.dl_set = 1
        self.total_to_dl = 0
        self.num_dl_done = 0
        self.amdownloading = False  # Variable to track download status

        # Create main layout as QHBoxLayout (horizontal layout)
        self.main_layout = QtWidgets.QHBoxLayout(self)

        # Left layout for the existing widgets
        self.left_layout = QtWidgets.QVBoxLayout()

        # Instructions Label
        self.instructions_label = QtWidgets.QLabel(self)
        instructions_text = (
            "Instructions:<br>"
            "1. Enter your Google Maps API key in the APIKEY field.<br>"
            "2. Choose 'Download from CSV' to download images based on coordinates from a CSV file, or 'Download Single' to download for a specific selected row.<br>"
            "3. Click 'Load CSV' to select your CSV file.<br>"
            "4. Select a row in the table to download a single image or start the bulk download process.<br>"
            "5. Use the 'Stop' button to halt the download process at any time.<br>"
            "6. The progress bar will show the current download status.<br>"
            "7. You can check the generated URL for each coordinate by clicking the 'What's this URL?' button."
        )
        self.instructions_label.setText(instructions_text)
        self.left_layout.addWidget(self.instructions_label)

        # Create a frame for mode selection
        self.mode_frame = QtWidgets.QGroupBox("Mode Selection")
        self.mode_layout = QtWidgets.QHBoxLayout()
        self.mode_frame.setLayout(self.mode_layout)

        self.dl_mode = QtWidgets.QButtonGroup(self)
        self.radio1 = QtWidgets.QRadioButton("Download from CSV")
        self.radio1.setChecked(True)
        self.dl_mode.addButton(self.radio1)
        self.mode_layout.addWidget(self.radio1)

        self.radio2 = QtWidgets.QRadioButton("Download Single")
        self.dl_mode.addButton(self.radio2)
        self.mode_layout.addWidget(self.radio2)

        self.left_layout.addWidget(self.mode_frame)

        self.L1 = QtWidgets.QLabel("APIKEY")
        self.left_layout.addWidget(self.L1)

        self.api_key_entry = QtWidgets.QLineEdit(self)
        self.api_key_entry.setPlaceholderText("Enter your API key")
        self.left_layout.addWidget(self.api_key_entry)

        self.download_directory_label = QtWidgets.QLabel("Download Directory")
        self.left_layout.addWidget(self.download_directory_label)

        self.download_directory_entry = QtWidgets.QLineEdit(self)
        self.download_directory_entry.setText(os.getcwd())  # Prefill with current working directory
        self.left_layout.addWidget(self.download_directory_entry)

        self.load_csv_button = QtWidgets.QPushButton("Load CSV", self)
        self.load_csv_button.clicked.connect(self.load_csv)
        self.left_layout.addWidget(self.load_csv_button)

        # Table
        self.tabledata = QtWidgets.QTableWidget(self)
        self.tabledata.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)  # Allow row selection
        self.left_layout.addWidget(self.tabledata)

        # Download
        self.download_button = QtWidgets.QPushButton("Download", self)
        self.download_button.clicked.connect(self.download)
        self.left_layout.addWidget(self.download_button)

        # Stop button
        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.stop_button.setEnabled(False)  # Initially disabled
        self.stop_button.clicked.connect(self.stop_download)
        self.left_layout.addWidget(self.stop_button)

        self.url_button = QtWidgets.QPushButton("What's this URL?", self)
        self.url_button.clicked.connect(self.check_url)
        self.left_layout.addWidget(self.url_button)

        # Progress
        self.progress_label = QtWidgets.QLabel("Progress:")
        self.left_layout.addWidget(self.progress_label)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setValue(0)
        self.left_layout.addWidget(self.progress_bar)

        # Text field for API response
        self.response_text = QTextEdit(self)
        self.response_text.setReadOnly(True)  # Make it read-only
        self.left_layout.addWidget(self.response_text)

        # Add left layout to the main layout
        self.main_layout.addLayout(self.left_layout)

        # Add a right pane with a green box for now
        self.right_pane = QFrame(self)
        self.right_pane.setFixedWidth(500)  # Set width of the pane
        self.right_pane.setStyleSheet("background-color: green;")  # Set background color to green

        # Add the right pane to the main layout
        self.main_layout.addWidget(self.right_pane)

        # Set the main layout
        self.setLayout(self.main_layout)


    def load_csv(self):
        csv_path = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV files (*.csv)")[0]
        if csv_path:
            try:
                self.csv_data = pd.read_csv(csv_path)
                QMessageBox.information(self, "CSV Loaded", f"CSV file loaded with {len(self.csv_data)} entries.")
                self.setup_table()
                self.update_table()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV file: {str(e)}")

    def setup_table(self):
        self.tabledata.setRowCount(len(self.csv_data))
        self.tabledata.setColumnCount(len(self.csv_data.columns))
        self.tabledata.setHorizontalHeaderLabels(self.csv_data.columns)

    def update_table(self):
        for i in range(len(self.csv_data)):
            for j in range(len(self.csv_data.columns)):
                self.tabledata.setItem(i, j, QTableWidgetItem(str(self.csv_data.iat[i, j])))

    def download_all(self):
        self.amdownloading = True
        self.stop_button.setEnabled(True)  # Enable stop button
        self.total_to_dl = len(self.csv_data)
        self.num_dl_done = 0
        self.progress_bar.setMaximum(self.total_to_dl)
        
        for i in range(self.total_to_dl):
            if not self.amdownloading:  # Check if download should stop
                break
            # Make API request
            current_row = self.csv_data.iloc[i]
            lat = current_row[0]
            long = current_row[1]
            key = self.api_key_entry.text()
            
            # Generate URL and perform request
            url = self.generate_streetview_url(lat, long, key)
            response = requests.get(url)

            # Display the response in the text field
            self.response_text.append(f"Response for {lat}, {long}:\n{response.text}\n")
            
            # Placeholder for actual download logic
            self.num_dl_done += 1
            self.progress_bar.setValue(self.num_dl_done)
            QtWidgets.QApplication.processEvents()  # Allow the GUI to update
        
        self.amdownloading = False
        self.stop_button.setEnabled(False)  # Disable stop button after download completes

    def download(self):
        if self.radio2.isChecked():  # Download Single
            current_row_index = self.tabledata.currentRow()
            if current_row_index != -1:  # Check if a row is selected
                # Retrieve the row data from the DataFrame
                row_data = self.csv_data.iloc[current_row_index]
                lat = row_data[0]
                long = row_data[1]
                key = self.api_key_entry.text()

                # Generate URL and perform request
                url = self.generate_streetview_url(lat, long, key)
                response = requests.get(url)

                # Display the response in the text field
                self.response_text.append(f"Response for {lat}, {long}:\n{response.text}\n")
            else:
                QMessageBox.warning(self, "Warning", "No row selected for download.")
        else:
            self.download_all()

    def check_url(self):
        current_row_index = self.tabledata.currentRow()
        if current_row_index != -1:  # Check if a row is selected
            # Retrieve the row data from the DataFrame
            row_data = self.csv_data.iloc[current_row_index]
            lat = row_data[0]
            long = row_data[1]
            key = self.api_key_entry.text() 
            res = self.generate_streetview_url(lat, long, key)
            print(lat, long, key)
            QMessageBox.information(self, "URL", res)
            print(row_data.to_string(index=False))  # Print the row data without the index
        else:
            print("No row is currently selected.")

    def generate_streetview_url(self, latitude, longitude, key, size="600x300", heading=0, pitch=0):
        """
        Generates a Google Street View API URL.

        Parameters:
        - latitude (float): Latitude of the location.
        - longitude (float): Longitude of the location.
        - key (str): Your Google Maps API key.
        - size (str): The size of the image (default is '600x300').
        - heading (int): The direction the camera is pointing (default is 0 degrees).
        - pitch (int): The angle of the camera tilt (default is 0 degrees).

        Returns:
        - str: The complete URL for the Street View API request.
        """
        base_url = "https://maps.googleapis.com/maps/api/streetview"
        url = f"{base_url}?size={size}&location={latitude},{longitude}&heading={heading}&pitch={pitch}&key={key}"
        return url

    def stop_download(self):
        self.amdownloading = False  # Set flag to stop download process

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = StreetViewDownloaderApp()
    window.show()
    sys.exit(app.exec_())