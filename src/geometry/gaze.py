"""
Track C: gaze as pitch / yaw (2 dimensions), extracted independently of track A.

Gaze is deliberately NOT read off the eyeLookUp / eyeLookDown / eyeLookIn /
eyeLookOut blendshape coefficients: those are a by-product of an expression
model rather than a gaze estimate, they are heavily discretised, and their
accuracy is not adequate for a signal the Dynamics layer will differentiate
three times. This module therefore owns its own estimator, selected through the
GazeBackend interface below.

Backends
--------
IrisGeometryBackend (default, zero extra dependencies)
    Geometric estimate from the iris landmarks (468-477) that the 478-point
    MediaPipe model already provides. This is NOT the rejected blendshape route
    -- it measures where the iris physically sits inside the eye opening -- but
    it is still a fallback: the mapping from iris offset to angle is an
    uncalibrated linear gain, so treat the output as a relative signal rather
    than an absolute gaze direction.
ExternalModelBackend
    Adapter for a dedicated appearance-based gaze model (L2CS-Net, ETH-XGaze and
    similar) loaded through onnxruntime. This is the intended production path;
    it needs weights plus a new dependency, i.e. an env-v0.2 bump, so the import
    is deferred and failure is reported with an explicit message instead of
    silently degrading.

Units are backend-dependent and are NOT interchangeable
-------------------------------------------------------
    ExternalModelBackend   real radians, from a model trained against calibrated
                           gaze ground truth.
    IrisGeometryBackend    a DIMENSIONLESS proxy: normalised iris displacement
                           (iris centre offset divided by eye width, roughly
                           within +/-0.5) multiplied by an uncalibrated gain. It
                           is monotonic in gaze angle and usable as a relative
                           signal, but it is not an angle and must never be
                           reported as degrees or radians.

Because the two backends emit different quantities, output from one is not a
valid training target alongside output from the other. build_backend() records
the active backend and its unit in geometry_metadata.json, and mixing runs is the
caller's responsibility to avoid.

Axis convention (both backends)
    index 0 = pitch, index 1 = yaw, in that order everywhere.
    Image coordinates increase downward and to the right, so a raw iris offset is
    positive downward. The iris backend negates the vertical component, making
    POSITIVE PITCH = LOOKING UP and POSITIVE YAW = LOOKING TOWARDS THE IMAGE
    RIGHT, which matches the usual convention and the sign an external model is
    expected to produce.

Post-processing (applied per clip in normalise_gaze)
    Zero-mean: each value has its clip-level mean over tracked frames subtracted,
    which removes the subject's habitual head-relative eye offset and any fixed
    camera-placement bias, leaving gaze motion rather than absolute direction.
    tanh compression: optional, off by default, and only meaningful for a backend
    whose output is in radians -- the limits it saturates at are physiological
    angles. Applying it to the dimensionless iris proxy would compare a ratio
    against a radian threshold, so normalise_gaze() refuses that combination
    rather than producing a number with no defensible meaning.
"""
import numpy as np

# Iris centres and eye corners in the 478-landmark topology. Left/right follow
# the convention already used by src/geometry/canonicalization.py.
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE = {"outer": 33, "inner": 133, "upper": 159, "lower": 145}
RIGHT_EYE = {"outer": 263, "inner": 362, "upper": 386, "lower": 374}

# Uncalibrated iris-offset -> angle gains, radians per unit of normalised offset.
IRIS_YAW_GAIN = 1.4
IRIS_PITCH_GAIN = 1.1
# Eye aspect ratio below which the eye is treated as closed and gaze undefined.
EYE_CLOSED_ASPECT_RATIO = 0.12

# Physiological limits used by the optional tanh compression, in radians.
PITCH_LIMIT_RAD = float(np.deg2rad(35.0))
YAW_LIMIT_RAD = float(np.deg2rad(50.0))

CONFIDENCE_METHOD = {
    "gaze": "quality proxy, not a probability: a native model score when the backend exposes one, otherwise (iris backend) a measured eye-aperture signal that falls to zero on blinks",
}

UNIT_RADIANS = "radians"
UNIT_IRIS_PROXY = "dimensionless_iris_offset_proxy"


class GazeBackend:
    """Interface every gaze estimator implements.

    estimate() receives the decoded RGB frame, the frame's normalised landmarks
    (478x3, or None when track A found no face) and the pixel image shape, and
    returns (pitch, yaw, confidence) with pitch first. The unit of pitch and yaw
    is declared by the subclass's `unit` attribute and is NOT assumed to be
    radians. It must return NaN and zero confidence rather than raising when it
    cannot produce an estimate.
    """

    name = "base"
    unit = None
    is_calibrated = False

    def estimate(self, frame_rgb, landmarks, image_shape) -> tuple:
        raise NotImplementedError

    def close(self) -> None:
        pass


class IrisGeometryBackend(GazeBackend):
    """Iris-offset geometry from the landmarks track A already computed.

    Output is a dimensionless proxy, not an angle. See the module docstring.
    """

    name = "iris-geometry (fallback, uncalibrated proxy)"
    unit = UNIT_IRIS_PROXY
    is_calibrated = False

    def estimate(self, frame_rgb, landmarks, image_shape) -> tuple:
        if landmarks is None or not np.isfinite(landmarks).all():
            return float("nan"), float("nan"), 0.0

        height, width = image_shape[0], image_shape[1]
        # Landmark x is px/width and y is px/height, so the normalised space is
        # anisotropic for any non-square frame. Undo that before measuring ratios.
        points = np.asarray(landmarks, dtype=np.float64)[:, :2] * np.array([width, height])

        offsets, apertures = [], []
        for eye, iris_index in ((LEFT_EYE, LEFT_IRIS_CENTER), (RIGHT_EYE, RIGHT_IRIS_CENTER)):
            outer, inner = points[eye["outer"]], points[eye["inner"]]
            upper, lower = points[eye["upper"]], points[eye["lower"]]
            iris = points[iris_index]

            eye_width = float(np.linalg.norm(outer - inner))
            eye_height = float(np.linalg.norm(upper - lower))
            if eye_width < 1e-6:
                continue
            aperture = eye_height / eye_width
            centre = (outer + inner) / 2.0
            delta = (iris - centre) / eye_width
            offsets.append(delta)
            apertures.append(aperture)

        if not offsets:
            return float("nan"), float("nan"), 0.0

        mean_offset = np.mean(offsets, axis=0)
        mean_aperture = float(np.mean(apertures))
        # Negated: image y grows downward, so an iris below the eye centre means
        # looking down. Positive pitch must mean looking up, as above.
        yaw = float(mean_offset[0] * IRIS_YAW_GAIN)
        pitch = float(-mean_offset[1] * IRIS_PITCH_GAIN)

        if mean_aperture <= EYE_CLOSED_ASPECT_RATIO:
            # Lids closed far enough that the iris position carries no gaze
            # information; report the geometry but with zero confidence.
            return float("nan"), float("nan"), 0.0
        confidence = float(min(1.0, mean_aperture / (2.0 * EYE_CLOSED_ASPECT_RATIO)))
        return pitch, yaw, confidence


class ExternalModelBackend(GazeBackend):
    """Dedicated appearance-based gaze model served through onnxruntime.

    Expects a model taking an NCHW RGB face crop and returning (pitch, yaw) in
    radians. Kept thin on purpose: swapping in a different architecture should
    only require changing the crop and the output ordering.
    """

    name = "external-onnx"
    unit = UNIT_RADIANS
    is_calibrated = True

    def __init__(self, model_path: str, input_size: int = 224, crop_margin: float = 0.25):
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError(
                "The external gaze backend needs onnxruntime, which is not part of frozen "
                "env-v0.1. Install it in an env-v0.2 environment, or run with "
                "--gaze_backend iris to stay within env-v0.1."
            ) from exc
        self._session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = input_size
        self._crop_margin = crop_margin
        self.name = "external-onnx:%s" % model_path.split("/")[-1]

    def estimate(self, frame_rgb, landmarks, image_shape) -> tuple:
        import cv2

        if landmarks is None or not np.isfinite(landmarks).all():
            return float("nan"), float("nan"), 0.0

        height, width = image_shape[0], image_shape[1]
        points = np.asarray(landmarks, dtype=np.float64)[:, :2] * np.array([width, height])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        margin_x = (x_max - x_min) * self._crop_margin
        margin_y = (y_max - y_min) * self._crop_margin
        x0 = int(max(0, x_min - margin_x))
        y0 = int(max(0, y_min - margin_y))
        x1 = int(min(width, x_max + margin_x))
        y1 = int(min(height, y_max + margin_y))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return float("nan"), float("nan"), 0.0

        crop = cv2.resize(frame_rgb[y0:y1, x0:x1], (self._input_size, self._input_size))
        tensor = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        pitch, yaw = np.asarray(self._session.run(None, {self._input_name: tensor})[0]).reshape(-1)[:2]
        return float(pitch), float(yaw), 1.0


def build_backend(kind: str, model_path: str = None) -> GazeBackend:
    if kind == "iris":
        return IrisGeometryBackend()
    if kind == "external":
        if not model_path:
            raise ValueError("--gaze_model_path is required when --gaze_backend external is used")
        return ExternalModelBackend(model_path)
    raise ValueError("unknown gaze backend: %r" % kind)


def normalise_gaze(pitch: np.ndarray, yaw: np.ndarray, confidence: np.ndarray,
                   apply_tanh: bool = False, unit: str = UNIT_RADIANS) -> tuple:
    """Zero-mean each axis over the clip, optionally tanh-compressing afterwards.

    Only frames with non-zero confidence contribute to the mean, so blinks and
    dropouts do not drag the baseline. Returns (pitch, yaw, reference) where
    reference is the subtracted (pitch_mean, yaw_mean).

    tanh compression saturates at PITCH_LIMIT_RAD / YAW_LIMIT_RAD, which are
    physiological ANGLES. Requesting it for a backend whose output is not in
    radians raises: dividing a dimensionless ratio by 0.87 rad and calling the
    result bounded would be arithmetic without meaning.
    """
    if apply_tanh and unit != UNIT_RADIANS:
        raise ValueError(
            "tanh compression saturates at radian limits but the active gaze backend emits %r; "
            "use a calibrated backend or leave compression off" % unit
        )
    pitch = np.asarray(pitch, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    usable = (np.asarray(confidence, dtype=np.float64) > 0.0) & np.isfinite(pitch) & np.isfinite(yaw)
    if not usable.any():
        return (pitch.astype(np.float32), yaw.astype(np.float32), np.full(2, np.nan))

    reference = np.array([pitch[usable].mean(), yaw[usable].mean()])
    pitch_centred = pitch - reference[0]
    yaw_centred = yaw - reference[1]

    if apply_tanh:
        pitch_centred = PITCH_LIMIT_RAD * np.tanh(pitch_centred / PITCH_LIMIT_RAD)
        yaw_centred = YAW_LIMIT_RAD * np.tanh(yaw_centred / YAW_LIMIT_RAD)

    return pitch_centred.astype(np.float32), yaw_centred.astype(np.float32), reference
