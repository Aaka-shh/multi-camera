import cv2
import hashlib
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ----------- CONFIGURABLES ------------

stream_urls = [
    0,  # Local webcam
      # Camera 1
      # Camera 2
    # Add more cameras if needed
]

CONF_THRESH = 0.3
MAX_DIST = 100
REID_INTERVAL = 10
FRAME_SCALE = 1.0
GRID_SIZE = 2  # 2x2 CCTV layout

# ----------- LOAD MODEL ------------

model = YOLO('yolov8l.pt')
product_labels = [name for idx, name in model.names.items() if name != 'person']

# ----------- ID GENERATOR ------------

class IDGen:
    def __init__(self, prefix):
        self.prefix = prefix
        self.counter = 1
        self.map = {}

    def get(self, f_hash):
        if f_hash not in self.map:
            self.map[f_hash] = f"{self.prefix}_{self.counter:03d}"
            self.counter += 1
        return self.map[f_hash]

# ----------- HASH EXTRACTOR ------------

def extract_hash(roi):
    if roi.size == 0:
        return "0"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    normalized_hist = cv2.normalize(hist, hist).flatten()
    return hashlib.md5(normalized_hist.tobytes()).hexdigest()

# ----------- DRAW FUNCTION ------------

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ----------- OPEN ALL CAMERA STREAMS ------------

caps = []
for url in stream_urls:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f" Failed to connect to stream: {url}")
        caps.append(None)
    else:
        print(f" Connected to stream: {url}")
        caps.append(cap)

# ----------- Initialize trackers and info per camera ------------

trackers = []
frame_counts = []
customer_gens = []
product_gens = []
customer_infos = []
product_infos = []
customer_product_maps = []

for _ in caps:
    trackers.append(DeepSort(max_age=30, n_init=3, nn_budget=50))
    frame_counts.append(0)
    customer_gens.append(IDGen("Customer"))
    product_gens.append(IDGen("Product"))
    customer_infos.append({})
    product_infos.append({})
    customer_product_maps.append({})

# ----------- MAIN LOOP ------------

try:
    while True:
        all_frames = []

        for i, cap in enumerate(caps):
            if cap is None:
                all_frames.append(None)
                continue

            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Failed to grab frame from camera {i}")
                all_frames.append(None)
                continue

            frame_counts[i] += 1
            frame_small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

            # Run YOLO detection
            results = model(frame_small, verbose=False)

            dets = []
            for det in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                label = model.names[int(cls)]
                if conf > CONF_THRESH:
                    dets.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, label))

            # DeepSORT tracking
            tracks = trackers[i].update_tracks(dets, frame=frame)

            for t in tracks:
                if not t.is_confirmed() or t.time_since_update > 1:
                    continue
                tid, bbox, label = t.track_id, t.to_ltrb(), t.get_det_class()
                roi = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                f_hash = extract_hash(roi) if frame_counts[i] % REID_INTERVAL == 0 else None

                if label == 'person':
                    if f_hash:
                        existing_id = customer_gens[i].map.get(f_hash)
                        if existing_id:
                            customer_infos[i][tid] = {'id': existing_id}
                        elif tid not in customer_infos[i]:
                            customer_infos[i][tid] = {'id': customer_gens[i].get(f_hash)}
                elif label in product_labels:
                    if tid not in product_infos[i] and f_hash:
                        product_infos[i][tid] = {'id': product_gens[i].get(f_hash), 'carrier': None}

            # Assign product to nearest customer
            for p_tid, p in product_infos[i].items():
                if p['carrier']:
                    continue
                pt = next((t for t in tracks if t.track_id == p_tid), None)
                if pt is None:
                    continue
                p_center = np.mean([[pt.to_ltrb()[0], pt.to_ltrb()[1]], [pt.to_ltrb()[2], pt.to_ltrb()[3]]], axis=0)
                for c_tid, c in customer_infos[i].items():
                    ct = next((t for t in tracks if t.track_id == c_tid), None)
                    if ct is None:
                        continue
                    c_center = np.mean([[ct.to_ltrb()[0], ct.to_ltrb()[1]], [ct.to_ltrb()[2], ct.to_ltrb()[3]]], axis=0)
                    dist = np.linalg.norm(p_center - c_center)
                    if dist < MAX_DIST:
                        p['carrier'] = c['id']
                        customer_product_maps[i].setdefault(c['id'], set()).add(p['id'])
                        break

            # Draw boxes
            for t in tracks:
                if not t.is_confirmed() or t.time_since_update > 1:
                    continue
                tid, bbox, label = t.track_id, t.to_ltrb(), t.get_det_class()
                if label == 'person' and tid in customer_infos[i]:
                    customer_id = customer_infos[i][tid]['id']
                    draw_box(frame, bbox, customer_id, (0, 255, 0))
                    product_count = len(customer_product_maps[i].get(customer_id, []))
                    cv2.putText(frame, f"Products: {product_count}", (int(bbox[0]), int(bbox[3]) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                elif tid in product_infos[i]:
                    color = (0, 0, 255) if product_infos[i][tid]['carrier'] else (255, 0, 0)
                    draw_box(frame, bbox, product_infos[i][tid]['id'], color)

            resized = cv2.resize(frame, (640, 360))
            all_frames.append(resized)

        # ----------- DISPLAY IN GRID (CCTV MONITOR STYLE) -----------
        rows = []
        row = []
        for i, f in enumerate(all_frames):
            if f is None:
                f = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(f, f"Camera {i} - Offline", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(f, f"Camera {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            row.append(f)
            if len(row) == GRID_SIZE:
                rows.append(np.hstack(row))
                row = []
        if row:
            while len(row) < GRID_SIZE:
                row.append(np.zeros((360, 640, 3), dtype=np.uint8))
            rows.append(np.hstack(row))

        full_display = np.vstack(rows)
        cv2.imshow("CCTV Multi-Camera Monitor", full_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    for cap in caps:
        if cap:
            cap.release()
    cv2.destroyAllWindows()

    print("\n--- Final Pickup Logs ---")
    for i, pickup_map in enumerate(customer_product_maps):
        print(f"Camera {i}:")
        for c, p in pickup_map.items():
            print(f"  {c} picked {len(p)} product(s): {', '.join(p)}")
