# ============================================================
# PHASE 3 - DISTANCE ESTIMATION
# ============================================================

def calculate_distance(
    focal_length,
    real_object_width_cm,
    object_width_pixels
):
    """
    Estimate distance using:

    Distance = (Focal Length × Real Object Width)
               / Object Width in Pixels
    """

    if object_width_pixels <= 0:
        return None

    distance_cm = (
        focal_length * real_object_width_cm
    ) / object_width_pixels

    return distance_cm


def get_distance_status(
    distance_cm,
    stop_distance_cm=30
):
    """
    Determine whether the target is far or close.
    """

    if distance_cm is None:
        return "UNKNOWN"

    if distance_cm <= stop_distance_cm:
        return "CLOSE"

    return "FAR"


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Temporary values for testing only.
    # These MUST be calibrated later.
    FOCAL_LENGTH = 500
    REAL_OBJECT_WIDTH_CM = 10
    OBJECT_WIDTH_PIXELS = 100

    distance = calculate_distance(
        FOCAL_LENGTH,
        REAL_OBJECT_WIDTH_CM,
        OBJECT_WIDTH_PIXELS
    )

    status = get_distance_status(
        distance,
        stop_distance_cm=30
    )

    print("DISTANCE TEST")
    print("-------------------------")

    if distance is not None:
        print(
            f"Estimated distance: "
            f"{distance:.2f} cm"
        )

    print(f"Status: {status}")