# ============================================================
# DISTANCE CALIBRATION
# ============================================================

print("======================================")
print(" ESP32-CAM DISTANCE CALIBRATION")
print("======================================")

print()

real_width_cm = float(
    input("Enter actual object width (cm): ")
)

known_distance_cm = float(
    input("Enter actual camera distance (cm): ")
)

object_width_pixels = float(
    input("Enter YOLO bounding-box width (pixels): ")
)


if real_width_cm <= 0:
    print("ERROR: Object width must be greater than 0.")
    exit()

if known_distance_cm <= 0:
    print("ERROR: Distance must be greater than 0.")
    exit()

if object_width_pixels <= 0:
    print("ERROR: Pixel width must be greater than 0.")
    exit()


focal_length = (
    object_width_pixels *
    known_distance_cm
) / real_width_cm


print()
print("======================================")
print(" CALIBRATION RESULT")
print("======================================")

print(
    f"Real object width : "
    f"{real_width_cm:.2f} cm"
)

print(
    f"Known distance    : "
    f"{known_distance_cm:.2f} cm"
)

print(
    f"Pixel width       : "
    f"{object_width_pixels:.2f} px"
)

print(
    f"Focal length      : "
    f"{focal_length:.2f}"
)

print("======================================")