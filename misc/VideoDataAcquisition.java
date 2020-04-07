package cz.vutbr.bpdataacquisition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Script;
import android.renderscript.Type;
import android.support.annotation.NonNull;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.view.Surface;
import android.widget.Toast;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class VideoDataAcquisition {

    // TAG
    private final String TAG = getClass().getSimpleName();

    // CONSTANTS
    private final int VIDEO_FRAME_RATE = 30;

    // FLAGS
    private boolean mIsRecording;

    // INNER CLASSES
    private final CameraCaptureSessionStateCallback mCameraCaptureSessionStateCallback = new CameraCaptureSessionStateCallback();
    private final CameraDeviceStateCallback mCameraDeviceStateCallback = new CameraDeviceStateCallback();
    private final CameraCaptureSessionCaptureCallback mCameraCaptureSessionCaptureCallback = new CameraCaptureSessionCaptureCallback();

    // OBJECTS
    private MediaRecorder mMediaRecorder;
    private CameraManager mCameraManager;
    private CameraDevice mCameraDevice;
    private ImageReader mImageReader;
    private CameraCharacteristics mCharacteristics;

    // VARIABLES
    private String mFilePath;
    private Size mVideoSize;
    private String mCameraId;

    // TIME STAMP
    private long firstFrameTime;
    private long captureStartTime;

    // THREADS
    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;

    // RENDERSCRIPT
    private RenderScript rs;

    // RENDERSCRIPT - VIDEO PROCESSING
    private Allocation rsYAlloc;
    private Allocation rsVAlloc;
    private Allocation rsRAlloc;
    private ScriptC_yuv2red rsYuv2Red;
    private Type.Builder rsTypeUcharY;
    private Type.Builder rsTypeUcharV;
    private Script.LaunchOptions rsLo;

    // CONTEXT
    private Context appContext;

    // LISTENERS
    private VideoListener listener;

    // CONSTRUCTOR
    public VideoDataAcquisition(Context _appContext) {
        Log.d(TAG, "INITIALIZING");
        appContext = _appContext;

        mIsRecording = false;

        // LISTENERS
        listener = null;

        setUpCameraManager();
        rs = RenderScript.create(appContext);
    }

    // LISTENERS - METHODS
    public interface VideoListener {
        public void onVideoValueChanged(float data);
    }

    // LISTENERS - SET UP
    public void setVideoListener(VideoListener _listener) {
        listener = _listener;
    }

    // TIME STAMP
    public long getFirstFrameTime() {
        return firstFrameTime;
    }

    public boolean isRecording() {
        return mIsRecording;
    }

    // THREADS - START (INITIALIZE)
    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("VideoThread");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    // THREADS - STOP (DESTROY)
    private void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            Log.d(TAG, "QUIT THREAD FAIL!" + e.getMessage());
        }
    }

    // CAMERA - SET UP CAMERA MANAGER
    private void setUpCameraManager() {
        mCameraManager = (CameraManager) appContext.getSystemService(appContext.CAMERA_SERVICE);
        try {
            // GET MAIN CAMERA
            mCameraId = mCameraManager.getCameraIdList()[0];

            // GET CHARACTERISTICS
            mCharacteristics = mCameraManager.getCameraCharacteristics(mCameraId);

            // CHECK FLASH LED
            if (!mCharacteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE)) {
                Toast.makeText(appContext, "Camera does not have flash unit!", Toast.LENGTH_LONG).show();
            }

        } catch (CameraAccessException e) {
            Log.d(TAG, "CAMERA NOT FOUND!" + e.getMessage());
        }
    }

    // VIDEO PROCESSING - SET UP IMAGE READER
    private void setUpImageReader() {
        mImageReader = ImageReader.newInstance(mVideoSize.getWidth(), mVideoSize.getHeight(), ImageFormat.YUV_420_888, 1);
        mImageReader.setOnImageAvailableListener(mOnImageAvailableListener, mBackgroundHandler);
    }

    // MEDIA RECORDER - SET UP
    private void setUpMediaRecorder() {
        try {
            mMediaRecorder = new MediaRecorder();

            // SOURCE
            mMediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);

            // FORMAT
            mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);

            // FILE PATH
            mMediaRecorder.setOutputFile(mFilePath);

            // FIND IDEAL VIDEO SIZE
            //CameraCharacteristics characteristics = mCameraManager.getCameraCharacteristics(mCameraId);
            StreamConfigurationMap map = mCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            mVideoSize = chooseVideoSize(map.getOutputSizes(MediaRecorder.class));

            // VIDEO
            mMediaRecorder.setVideoFrameRate(VIDEO_FRAME_RATE);
            mMediaRecorder.setVideoSize(mVideoSize.getWidth(), mVideoSize.getHeight());
            mMediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);

            // ORIENTATION
            mMediaRecorder.setOrientationHint(90);

            // PREPARE MEDIA RECORDER
            mMediaRecorder.prepare();
        } catch (IOException e) {
            Log.d(TAG, "SET UP MEDIA RECORDER FAIL!" + e.getMessage());
        } //catch (CameraAccessException e) {
//            Log.d(TAG, "CANNOT ACCESS THE CAMERA!" + e.getMessage());
//        }
    }

    // CAMERA - CHOOSE VIDEO SIZE
    private Size chooseVideoSize(Size[] choices) {
        for (Size size : choices) {
            if (size.getWidth() == size.getHeight() * 4 / 3 && size.getWidth() <= 400) {
                return size;
            }
        }
        Log.d(TAG, "Couldn't find any suitable video size");
        return choices[choices.length - 1];
    }

    // PUBLIC METHOD - START CAPTURE
    public void startRecording(String filePath) {
        Log.d(TAG, "START HIT");
        mFilePath = filePath + ".mp4";

        firstFrameTime = 0;
        list = new ArrayList<VideoTimeStamp>();

        startBackgroundThread();
        setUpMediaRecorder();
        setUpRsYuv2Red();

        try {
            mCameraManager.openCamera(mCameraId, mCameraDeviceStateCallback, mBackgroundHandler);
        } catch (CameraAccessException | SecurityException e) {
            Log.d(TAG, "CANNOT ACCESS THE CAMERA!" + e.getMessage());
        }
    }

    // PUBLIC METHOD - STOP CAPTURE
    public void stopRecording() {
        Log.d(TAG, "STOP HIT");

        mIsRecording = false;

        mMediaRecorder.stop();
        mMediaRecorder.reset();
        mMediaRecorder.release();
        mMediaRecorder = null;

        mCameraDevice.close();

        stopBackgroundThread();
    }

    // CAPTURE CALLBACK
    private class CameraCaptureSessionCaptureCallback extends CameraCaptureSession.CaptureCallback {
        @Override
        public void onCaptureStarted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, long timestamp, long frameNumber) {
            super.onCaptureStarted(session, request, timestamp, frameNumber);
            if (frameNumber == 0) cas3 = SystemClock.elapsedRealtimeNanos();
        }

        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull TotalCaptureResult result) {
            super.onCaptureCompleted(session, request, result);
        }
    }

    // CAPTURE STATE CALLBACK
    private class CameraCaptureSessionStateCallback extends CameraCaptureSession.StateCallback {
        @Override
        public void onReady(@NonNull CameraCaptureSession session) {
            Log.d(TAG, "CameraCaptureSessionStateCallback: onReady()");
            super.onReady(session);
            try {
                CaptureRequest.Builder builder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);

                // CAMERA SETTING
                builder.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_TORCH);

                // CAMERA CAPTURE CONTROL
                builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF);
                builder.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF);
                builder.set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_OFF);


                mCharacteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE);

                // SURFACES ADD
                builder.addTarget(mImageReader.getSurface());
                builder.addTarget(mMediaRecorder.getSurface());

                // CREATE CAPTURE REQUEST
                CaptureRequest request = builder.build();

                // START SESSION
                session.setRepeatingRequest(request, mCameraCaptureSessionCaptureCallback, mBackgroundHandler);
                cas2 = SystemClock.elapsedRealtimeNanos();

                // START RECORDING WITH MEDIA RECORDER
                mMediaRecorder.start();
            } catch (CameraAccessException e) {
                Log.d(TAG, "CameraCaptureSessionStateCallback: onReady()" + e.getMessage());
            }
        }

        @Override
        public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
            Log.d(TAG, "CameraCaptureSessionStateCallback: onConfigured()");
        }

        @Override
        public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
            Log.d(TAG, "CameraCaptureSessionStateCallback: onConfigureFailed()");
        }

        @Override
        public void onActive(@NonNull CameraCaptureSession session) {
            Log.d(TAG, "CameraCaptureSessionStateCallback: onActive()");
            mIsRecording = true;
            cas4 = SystemClock.elapsedRealtimeNanos();
            super.onActive(session);
        }

        @Override
        public void onClosed(@NonNull CameraCaptureSession session) {
            Log.d(TAG, "CameraCaptureSessionStateCallback: onClosed()");
            super.onClosed(session);
        }

        @Override
        public void onCaptureQueueEmpty(@NonNull CameraCaptureSession session) {
            Log.d(TAG, "CameraCaptureSessionStateCallback: onCaptureQueueEmpty()");
            super.onCaptureQueueEmpty(session);
        }

        @Override
        public void onSurfacePrepared(@NonNull CameraCaptureSession session, @NonNull Surface surface) {
            Log.d(TAG, "CameraCaptureSessionStateCallback: onSurfacePrepared()");
            super.onSurfacePrepared(session, surface);
        }
    }

    // DEVICE STATE CALLBACK
    private class CameraDeviceStateCallback extends CameraDevice.StateCallback {
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            Log.d(TAG, "CameraDeviceStateCallback: onOpened()");
            mCameraDevice = cameraDevice;
            setUpImageReader();
            try {
                // GET SURFACES
                List<Surface> surfaces = new ArrayList<>();
                surfaces.add(mMediaRecorder.getSurface());
                surfaces.add(mImageReader.getSurface());
                cameraDevice.createCaptureSession(surfaces, mCameraCaptureSessionStateCallback, mBackgroundHandler);
            } catch (CameraAccessException e) {
                Log.d(TAG, "CameraDeviceStateCallback: onOpened()" + e.getMessage());
            }
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            Log.d(TAG, "CameraDeviceStateCallback: onDisconnected()");
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int i) {
            Log.d(TAG, "CameraDeviceStateCallback: onError()");
        }

        @Override
        public void onClosed(@NonNull CameraDevice camera) {
            Log.d(TAG, "CameraDeviceStateCallback: onClosed()");
            super.onClosed(camera);
        }
    }

    // VIDEO PROCESSING
    public long cas1;
    public long cas2;
    public long cas3;
    public long cas4;
    private final ImageReader.OnImageAvailableListener mOnImageAvailableListener = new ImageReader.OnImageAvailableListener() {
        @Override
        public void onImageAvailable(ImageReader reader) {
            Image image =  reader.acquireNextImage();

            // JPEG IMAGE TO BITMAP - LOW FPS - NOT USED!
//            ByteBuffer buffer = image.getPlanes()[0].getBuffer();
//            byte[] bytes = new byte[buffer.capacity()];
//            buffer.get(bytes);
//            Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, null);
//            int redAvg = getAverageValueOfRedChannel(bitmap);

            // TESTING
//            long time = System.nanoTime();
//            long cas = image.getTimestamp();

//

            long time = image.getTimestamp();
            if (firstFrameTime == 0) {
                firstFrameTime = time;
                cas1 = SystemClock.elapsedRealtimeNanos();
            }

            float redAvg = getAverageValueOfRedChannel(image);

            // TESTING
            list.add(new VideoTimeStamp(redAvg, time));

            if(listener != null) {
                listener.onVideoValueChanged(redAvg);
            }
            image.close();
        }
    };

    // Red average value for JPEG processing - NOT USED!
    private int getAverageValueOfRedChannel(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        int red = 0;

        for (int x = 0; x < width; x++)
            {
            for (int y = 0; y < height; y++)
                {
                int pixel = bitmap.getPixel(x, y);
                red += (pixel >> 16) & 0xff;
            }
        }
        return red / (width * height);
    }

    private void setUpRsYuv2Red() {
        rsYuv2Red = new ScriptC_yuv2red(rs);
        rsTypeUcharY = new Type.Builder(rs, Element.U8(rs));
        rsTypeUcharV = new Type.Builder(rs, Element.U8(rs));
        rsRAlloc = Allocation.createSized(rs, Element.F32(rs), 1);
        rsLo = new Script.LaunchOptions();
    }

    private float getAverageValueOfRedChannel(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Get the three image planes
        Image.Plane[] planes = image.getPlanes();

        ByteBuffer buffer = planes[0].getBuffer();
        byte[] y = new byte[buffer.remaining()];
        buffer.get(y);

        buffer = planes[2].getBuffer();
        byte[] v = new byte[buffer.remaining()];
        buffer.get(v);

        int yRowStride = planes[0].getRowStride(); // 320
        int vRowStride = planes[1].getRowStride(); // 320
        int vPixelStride = planes[1].getPixelStride(); // 2

        // SET Y
        rsTypeUcharY.setX(yRowStride);
        rsTypeUcharY.setY(height);
        rsYAlloc = Allocation.createTyped(rs, rsTypeUcharY.create());
        rsYAlloc.copyFrom(y);
        rsYuv2Red.set_ypsIn(rsYAlloc);

        // SET V
        rsTypeUcharV.setX(v.length);
        rsVAlloc = Allocation.createTyped(rs, rsTypeUcharV.create());
        rsVAlloc.copyFrom(v);
        rsYuv2Red.set_vIn(rsVAlloc);

        // SET PARAMETERS
        rsYuv2Red.set_vRowStride(vRowStride);
        rsYuv2Red.set_vPixelStride(vPixelStride);
        rsYuv2Red.set_width(width);
        rsYuv2Red.set_height(height);

        // LO
        rsLo.setX(0, width);  // by this we ignore the y’s padding zone, i.e. the right side of x between width and yRowStride
        rsLo.setY(0, height);

        rsYuv2Red.invoke_reset();
        rsYuv2Red.forEach_getRedFromYuv(rsLo);
        rsYuv2Red.forEach_getRedAvg(rsRAlloc);

        float average[] = new float[1];
        rsRAlloc.copyTo(average);
        return average[0];
    }

    // TESTING
    public List<VideoTimeStamp> list;

    public class VideoTimeStamp {
        VideoTimeStamp(float red, long time) {
            avgRedValue = red;
            frameTimeStamp = time;
        }

        public float avgRedValue;
        public long frameTimeStamp;
    }
}
