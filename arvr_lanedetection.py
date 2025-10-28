import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import time
from natsort import natsorted 
from datetime import datetime 


snapshot_folder = r"C:\Users\ojasr\OneDrive\Desktop\arvr\Assets\Snapshots"

graph_save_folder = r"C:\Users\ojasr\OneDrive\Desktop\arvr\steering_offset_graphs"


if not os.path.exists(graph_save_folder):
    os.makedirs(graph_save_folder)
    print(f"Created graph save folder: {graph_save_folder}")


snapshot_files = [os.path.join(snapshot_folder, f) for f in os.listdir(snapshot_folder) if f.endswith(".png")]
snapshot_files = natsorted(snapshot_files)

if not snapshot_files:
    print("No snapshots found in folder:", snapshot_folder)
    exit()


def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

offset_values = []
xm_per_pix = 3.7/700 # meters per pixel in x


fig, ax = plt.subplots(figsize=(8, 4)) 
line, = ax.plot([], 'r-')
ax.set_ylim([-2,2])
ax.set_xlabel('Frame')
ax.set_ylabel('Offset (m)')
ax.set_title('Car Offset')
ax.grid(True) # Added grid
fig.tight_layout()
canvas = FigureCanvas(fig)


for snapshot_path in snapshot_files:
    image = cv2.imread(snapshot_path)
    frame = cv2.resize(image, (640,480))

    # --- Perspective transform (zoomed out) ---
    tl, bl, tr, br = (222,387), (70,472), (400,380), (538,472)
    pts1 = np.float32([tl, bl, tr, br])

    
    dst_width = 800 
    dst_height = 480
    pts2 = np.float32([
        [dst_width*0.2, 0], # top-left
        [dst_width*0.2, dst_height], # bottom-left
        [dst_width*0.8, 0], # top-right
        [dst_width*0.8, dst_height] # bottom-right
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (dst_width,dst_height))

    # --- HSV thresholding ---
    hsv_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_frame, lower, upper)

    # --- Sliding window histogram ---
    histogram = np.sum(mask[mask.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    y = dst_height
    msk = mask.copy()
    left_pts = []
    right_pts = []

    while y > 0:
        # Left window
        img = mask[y-40:y, max(left_base-50,0):left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                left_base = left_base-50 + cx
        cv2.rectangle(msk, (max(left_base-50,0),y), (left_base+50,y-40), (255,255,255), 2)
        left_pts.append((left_base,y))

        # Right window
        img = mask[y-40:y, max(right_base-50,0):right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                right_base = right_base-50 + cx
        cv2.rectangle(msk, (max(right_base-50,0),y), (right_base+50,y-40), (255,255,255), 2)
        right_pts.append((right_base,y))

        y -= 40

    # --- Overlay lane ---
    if left_pts and right_pts:
        left_pts_arr = np.array(left_pts, np.int32)
        right_pts_arr = np.array(right_pts, np.int32)
        lane_pts = np.vstack((left_pts_arr, right_pts_arr[::-1]))
        cv2.fillPoly(transformed_frame, [lane_pts], (0,255,0))
        cv2.polylines(transformed_frame, [lane_pts], True, (0,0,255), 2)

    # --- Car offset ---
    lane_center = (left_base + right_base)/2
    car_center = dst_width/2
    offset_pix = car_center - lane_center
    offset_m = offset_pix * xm_per_pix
    offset_values.append(offset_m)
    if len(offset_values) > 100:
        offset_values = offset_values[-100:]

    # --- Update Matplotlib graph (LIVE DISPLAY - NO CHANGE) ---
    line.set_ydata(offset_values)
    line.set_xdata(range(len(offset_values)))
    ax.relim()
    ax.autoscale_view()
    canvas.draw()
    buf, (w,h) = canvas.print_to_buffer()
    graph_img = np.frombuffer(buf, np.uint8).reshape((h,w,4))
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
    graph_img = cv2.resize(graph_img, (320,240))

    # --- Collage (LIVE DISPLAY - NO CHANGE) ---
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    msk_bgr = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)

    resized_transformed = cv2.resize(transformed_frame, (640, 480))
    resized_mask_bgr = cv2.resize(mask_bgr, (640, 480))
    resized_msk_bgr = cv2.resize(msk_bgr, (640, 480))

    top_row = cv2.hconcat([frame, resized_transformed])
    bottom_row = cv2.hconcat([resized_mask_bgr, resized_msk_bgr])
    collage = cv2.vconcat([top_row, bottom_row])

    h_g, w_g, _ = graph_img.shape
    collage[0:h_g, collage.shape[1]-w_g:collage.shape[1]] = graph_img

    cv2.putText(collage, f"Offset: {offset_m:.2f} m", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Lane Detection + Live Graph", collage)

    # --- Small delay to simulate frame rate ---
    if cv2.waitKey(50) & 0xFF == 27:
        break

cv2.destroyAllWindows()


if offset_values:
    # 1. Update the final graph with all data points
    ax.clear()
    ax.plot(offset_values, 'r-')
    ax.set_ylim([-2, 2])
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Offset (m)')
    ax.set_title(f'Final Car Offset (Total Frames: {len(offset_values)})')
    ax.grid(True)
    fig.tight_layout()

    # 2. Generate a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"steering_offset_graph_{timestamp}.png"
    save_path = os.path.join(graph_save_folder, filename)

    # 3. Save the figure
    fig.savefig(save_path)
    print(f"\n✅ Final steering offset graph saved to: {save_path}")
else:
    print("\n⚠️ No offset data was recorded, graph not saved.")

plt.close(fig) # Close the figure to free up memory