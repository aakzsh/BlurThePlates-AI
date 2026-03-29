## BlurThePlates

BlurThePlates is a specialized video processing API system designed to automatically detect and obscure license plates in driving footage. It utilizes a YOLOv8-based computer vision pipeline to identify plates and applies a temporal tracking and pixelation algorithm to ensure privacy even when detections are momentarily lost.

### Background and Purpose
Automated driving footage (dashcams) often captures sensitive PII (Personally Identifiable Information) in the form of vehicle license plates. Standard frame-by-frame detection often suffers from "flicker" where a plate is visible for a single frame due to motion blur or lighting, bypassing simple filters.

This project solves this by implementing a **PlateTracker** with memory persistence. If a plate is detected in Frame A but missed in Frame B, the system "remembers" the last known position and continues to apply the blur for a set number of frames, ensuring 100% privacy coverage.

---

### Technical Implementation

#### 1. Detection Engine
The system uses the YOLOv8 architecture with a custom-trained model (`plate_detector.pt`).
* **Tracking:** It uses the ByteTrack algorithm (`persist=True`) to maintain unique IDs for every detected plate across frames.

#### 2. Temporal Smoothing & Persistence
The `PlateTracker` class manages the state of detected objects:
* **Box Smoothing:** It applies a weighted average ($70\%$ previous, $30\%$ current) to the bounding box coordinates to prevent the blur area from "shaking."
* **Memory Frames:** A `life` counter is assigned to each track ID. If the model fails to detect a plate in a frame, the tracker continues to return the last known box until the `life` counter reaches zero.

#### 3. Pixelation Filter
Instead of a simple black box, the system applies a professional pixelation effect:
* **ROI Expansion:** The bounding box is padded by 12 pixels on all sides to ensure the edges of the plate are covered.
* **Resizing:** The region is downsampled to a 16x16 grid and then scaled back up using `INTER_NEAREST` interpolation to create the blocky privacy effect.

---

### API Endpoints

The Flask application exposes two main endpoints used for automation pipelines:

* **POST `/upload`**
    * Accepts a video file under the key `video`.
    * Generates a unique UUID for the session.
    * Triggers the `process_video` function.
    * Returns a JSON response containing a `download_url`.

* **GET `/download/<file_id>`**
    * Retrieves the processed `.mp4` file from the output directory.

---

### Local Setup

#### Prerequisites
* Python 3.10+
* FFmpeg (for video encoding)

#### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install ultralytics flask opencv-python numpy
   ```
3. Ensure the following directory structure exists:
   ```text
   /BlurThePlates
   ├── uploads/
   ├── outputs/
   ├── plate_detector.pt
   └── app.py
   ```

#### Running the Server
```bash
python app.py
```
The server will start on `http://localhost:5000`.

---

### Integration with n8n
This project is designed to act as a worker node in an n8n automation flow:
1.  **Google Drive Trigger:** Detects new footage in a specific folder.
2.  **HTTP Request (POST):** Sends the file to the `/upload` endpoint.
3.  **Wait Node:** Allows time for the GPU to process the frames.
4.  **HTTP Request (GET):** Downloads the resulting "blurred" video.
5.  **YouTube Node:** Uploads the final processed video to the channel.
