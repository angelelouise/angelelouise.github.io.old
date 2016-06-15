package com.example.victor.laface;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.OrientationEventListener;
import android.view.Window;
import android.view.WindowManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class FAceDetection extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "App";
    private CascadeClassifier cascade, mCascadeER, mCascadeNariz, mCascadeBoca;
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgba, mGray;
    private boolean mIsFrontCamera = false;
    // Used in Camera selection from menu (when implemented)
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;
    private OrientationEventListener orientationListener = null;

    private static final int VIEW_MODE_KLT_TRACKER = 0;
    private static final int VIEW_MODE_OPTICAL_FLOW = 1;
    private static final int OLHOS = 1;
    private static final int BOCA = 2;
    private static final int NARIZ = 3;
    org.opencv.core.Rect[] facesArray;
    MatOfRect faces;
    private boolean achouFace;
    private int nFaces;
    private ArrayList<Rect> listaRetangulos;
    private int countFrames;

    private int mViewMode;
    private Mat mPrevGray;
    MatOfPoint2f prevFeatures, nextFeatures;
    MatOfPoint features;
    MatOfByte status;
    MatOfFloat err;
    ArrayList<Point> pontos;
    private MenuItem mItemPreviewOpticalFlow, mItemPreviewKLT;
    int width;
    int height;
    private Rect nariz;

    private BaseLoaderCallback mLoaderCallback = new
            BaseLoaderCallback(this) {
                @Override
                public void onManagerConnected(int status) {
                    switch (status) {
                        case LoaderCallbackInterface.SUCCESS:
                            //Log.i(TAG, "OpenCV sucess");
                            try {

                                // --------------------------------- load face classificator ------------------------------------
                                InputStream is = getResources().openRawResource(R.raw.lbp_cascadefrontal);
                                File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                                File mCascadeFile = new File(cascadeDir, "lbp_cascadefrontal.xml");
                                FileOutputStream os = new FileOutputStream(mCascadeFile);
                                byte[] buffer = new byte[4096];
                                int bytesRead;
                                while ((bytesRead = is.read(buffer)) != -1) {
                                    os.write(buffer, 0, bytesRead);
                                }
                                is.close();
                                os.close();
                                //----------------------------------------------------------------------------------------------------

                                // --------------------------------- load eye classificator -----------------------------------
                                InputStream iser = getResources().openRawResource(R.raw.haarcascade_eye);
                                File cascadeDirER = getDir("cascadeER", Context.MODE_PRIVATE);
                                File cascadeFileER = new File(cascadeDirER, "haarcascade_mcs_eyepair_small.xml");
                                FileOutputStream oser = new FileOutputStream(cascadeFileER);

                                byte[] bufferER = new byte[4096];
                                int bytesReadER;
                                while ((bytesReadER = iser.read(bufferER)) != -1) {
                                    oser.write(bufferER, 0, bytesReadER);
                                }
                                iser.close();
                                oser.close();
                                //----------------------------------------------------------------------------------------------------

                                // --------------------------------- load nose classificator ------------------------------------
                                InputStream isel = getResources().openRawResource(R.raw.haarcascade_mcs_nose);
                                File cascadeDirNariz = getDir("mCascadeNariz", Context.MODE_PRIVATE);
                                File cascadeFileNariz = new File(cascadeDirNariz, "haarcascade_mcs_nose.xml");
                                FileOutputStream osel = new FileOutputStream(cascadeFileNariz);

                                byte[] bufferEL = new byte[4096];
                                int bytesReadEL;
                                while ((bytesReadEL = isel.read(bufferEL)) != -1) {
                                    osel.write(bufferEL, 0, bytesReadEL);
                                }
                                isel.close();
                                osel.close();

                                // ------------------------------------------------------------------------------------------------------

                                // --------------------------------- load mouth classificator ------------------------------------
                                InputStream ismo = getResources().openRawResource(R.raw.haarcascade_mcs_mouth);
                                File cascadeBoca = getDir("mCascadeBoca", Context.MODE_PRIVATE);
                                File cascadeFileBoca = new File(cascadeBoca, "haarcascade_mcs_mouth.xml");
                                FileOutputStream osmo = new FileOutputStream(cascadeFileBoca);

                                byte[] bufferMo = new byte[4096];
                                int bytesReadMo;
                                while ((bytesReadMo = ismo.read(bufferMo)) != -1) {
                                    osmo.write(bufferMo, 0, bytesReadMo);
                                }
                                ismo.close();
                                osmo.close();

                                // ------------------------------------------------------------------------------------------------------
                                cascade = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                                mCascadeER = new CascadeClassifier(cascadeFileER.getAbsolutePath());
                                mCascadeNariz = new CascadeClassifier(cascadeFileNariz.getAbsolutePath());
                                mCascadeBoca= new CascadeClassifier(cascadeFileBoca.getAbsolutePath());
                                faces = new MatOfRect();
                                if (cascade.empty() || mCascadeER.empty() || mCascadeNariz.empty()|| mCascadeBoca.empty()) {
                                    Log.i("Cascade Error", "Failed to loadcascade classifier");
                                    cascade = null;
                                }

                            } catch (Exception e) {
                                Log.i("Cascade Error: ", "Cascase not found");
                            }
                            mOpenCvCameraView.enableView();
                            break;
                        default: {
                            super.onManagerConnected(status);
                        }
                        break;
                    }
                }
            };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        achouFace = false;
        super.onCreate(savedInstanceState);
        final Window window = getWindow();
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_face_detection);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        this.height = height;
        this.width = width;
        resetVars();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.
                OPENCV_VERSION_2_4_10, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mGray = inputFrame.gray();
        mRgba = inputFrame.rgba();

        listaRetangulos = new ArrayList<Rect>();

        if (!achouFace) {
            nFaces=0;
            pontos = new ArrayList<Point>();
            if (cascade != null) {
                cascade.detectMultiScale(mGray, faces, 1.15, 3, 1, new Size(100, 100), new Size(400, 400));
            }
            facesArray = faces.toArray();
            if (facesArray.length > 0 && countFrames < 7) {
                nFaces++;
                for (int i = 0; i < facesArray.length; i++) {
                    adicionaPontos(pontos, facesArray[i]);
                    listaRetangulos.add(facesArray[i]);
                    Mat retanguloGray=
                    Imgproc.eq
                    achaObjetosPorClassificador(mGray, redimensionaROI(facesArray[i], OLHOS), mCascadeER, pontos, OLHOS, 20, 20);
                    achaObjetosPorClassificador(mGray, redimensionaROI(facesArray[i], NARIZ), mCascadeNariz, pontos, NARIZ, 20, 30);
                    achaObjetosPorClassificador(mGray, redimensionaROI(facesArray[i], BOCA), mCascadeBoca, pontos, BOCA, 20, 30);
                }
                desenhaRetangulos(listaRetangulos);
                desenhaPontos(pontos);
                countFrames++;
            }
            if (countFrames >= 7) {
                achouFace = true;
            }
            texto(mRgba,new Point(30,50),"Aprendendo", 1.2,new Scalar(255));
        }

        if (achouFace) {
            if (pontos.size() < nFaces*8) {
                countFrames = 0;
                achouFace = false;
            } else {

                if (features.toArray().length == 0) {
                    Point points[] = new Point[pontos.size()];
                    pontos.toArray(points);
                    features.fromArray(points);
                    prevFeatures.fromList(features.toList());
                    mPrevGray = mGray.clone();
                }

                Video.calcOpticalFlowPyrLK(mPrevGray, mGray,
                        prevFeatures, nextFeatures, status, err);

                List<Point> prevList = prevFeatures.toList(),
                        nextList = nextFeatures.toList();
                Scalar color = new Scalar(255);
                MatOfPoint tempNext;
                for (int i = 0; i < prevList.size(); i++) {
                    Core.line(mRgba, prevList.get(i), nextList.get(i), color, 4);
                    Core.circle(mRgba, prevList.get(i), 5, new Scalar(255));
                    Core.circle(mRgba, nextList.get(i), 5, new Scalar(0, 255));

                    tempNext = new MatOfPoint();
                    nextFeatures.convertTo(tempNext, CvType.CV_32S);
                    List<MatOfPoint> contourTemp = new ArrayList<>();
                    contourTemp.add(tempNext);

                }
                checkPoints(nextList);
                mPrevGray = mGray.clone();
                prevFeatures.fromArray(nextFeatures.toArray());
                //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Lucas-kanade<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            }
        }
        return mRgba;
    }

    private void achaObjetosPorClassificador(Mat mGray, Rect imageRect, CascadeClassifier classificador, ArrayList<Point> pontos, int option, int sizex, int sizey) {
        Mat subMat = mGray.submat(imageRect);
        Rect newReducedRet;
        Rect novoRet;
        //Mat mROI;
        org.opencv.core.Rect[] facesArray;
        MatOfRect identificado = new MatOfRect();
        int elementosInseridos = 0;
        switch (option) {
            case (OLHOS):
                if (classificador != null) {
                    classificador.detectMultiScale(subMat, identificado, 1.10, 5, 1, new Size(sizex, sizey), new Size(imageRect.width, imageRect.height));
                }
                facesArray = identificado.toArray();
                for (int i = 0; i < facesArray.length; i++) {
                    if (elementosInseridos < 2) {
                        //reduz o retângulo achado dos olhos
                        newReducedRet = reducaoRet(facesArray[i], 0, 0.2);
                        novoRet = novaRef(imageRect, newReducedRet);
                        listaRetangulos.add(novoRet);
                        adicionaPontos(pontos, new Point(novoRet.x + (int) (novoRet.width / 2), novoRet.y + (int) (novoRet.height / 2)));
                        elementosInseridos++;
                    }
                }
                break;
            default:

                if (classificador != null) {
                    classificador.detectMultiScale(subMat, identificado, 1.15, 3, 1, new Size(sizex, sizey), new Size(imageRect.width, imageRect.height));
                }
                facesArray = identificado.toArray();
                for (int i = 0; i < facesArray.length; i++) {
                    if (elementosInseridos < 1) {
                        newReducedRet = facesArray[i];
                        novoRet = novaRef(imageRect, newReducedRet);
                        listaRetangulos.add(novoRet);
                        adicionaPontos(pontos, new Point(novoRet.x + (int) (novoRet.width / 2), novoRet.y + (int) (novoRet.height / 2)));
                        elementosInseridos++;
                    }
                }
                break;
        }
    }

    private void resetVars() {
        mPrevGray = new Mat(mRgba.rows(), mRgba.cols(), CvType.
                CV_8UC1);
        features = new MatOfPoint();
        prevFeatures = new MatOfPoint2f();
        nextFeatures = new MatOfPoint2f();
        status = new MatOfByte();
        err = new MatOfFloat();
        countFrames = 0;
    }

    private void checkPoints(List<Point> pontos) {
        for (Point p : pontos) {
            if (p.x >= width || p.y >= height || p.x <= 0 || p.y <= 0) {
                achouFace = false;
                resetVars();
                break;
            }
        }
    }

    private void texto(Mat img, Point p, String mens, double scale, Scalar cor) {

        Core.putText(mRgba,
                mens,
                p,
                Core.FONT_HERSHEY_COMPLEX,
                scale,
                cor);
    }

    private void adicionaPontos(ArrayList<Point> pontos, Rect ret) {
        pontos.add(new Point(ret.tl().x, ret.tl().y));
        pontos.add(new Point(ret.tl().x + ret.width, ret.tl().y));
        pontos.add(new Point(ret.br().x, ret.br().y));
        pontos.add(new Point(ret.tl().x, ret.tl().y + ret.height));
    }

    private void adicionaPontos(ArrayList<Point> pontos, Point ponto) {
        pontos.add(ponto);
    }

    private void desenhaRetangulos(ArrayList<Rect> retangulos) {
        int c = 0;
        for (Rect p : retangulos) {
            Core.rectangle(mRgba, p.tl(), p.br(), new Scalar(255), 5);
            //horizontais
            if (c == 0) {
                Core.line(mRgba, new Point(p.x, (int) (p.y + p.height * 0.3)),
                        new Point((p.x + p.width), (int) (p.y + p.height * 0.3)), new Scalar(0, 0, 255));
                Core.line(mRgba, new Point(p.x, (int) (p.y + p.height * 0.6)),
                        new Point((p.x + p.width), (int) (p.y + p.height * 0.6)), new Scalar(0, 0, 255));

                //verticais
                Core.line(mRgba, new Point((int) (p.x + p.width * 0.3), p.y),
                        new Point((int) (p.x + p.width * 0.3), (p.y + p.height)), new Scalar(0, 0, 255));
                Core.line(mRgba, new Point((int) (p.x + p.width * 0.6), p.y),
                        new Point((int) (p.x + p.width * 0.6), (p.y + p.height)), new Scalar(0, 0, 255));
                c++;
            }
        }
    }

    private void desenhaPontos(ArrayList<Point> pontos) {
        for (Point p : pontos)
            Core.circle(mRgba, p, 4, new Scalar(0, 255));
    }

    private Rect novaRef(Rect refMaior, Rect refMenor) {
        Point ptl = new Point(refMaior.tl().x + refMenor.tl().x,
                refMaior.tl().y + refMenor.tl().y);

        Point pbr = new Point(refMaior.tl().x + refMenor.br().x,
                refMaior.tl().y + refMenor.br().y);
        return new Rect(ptl, pbr);
    }

    //Redução do retnagulo proporcionalmente
    private Rect reducaoRet(Rect ret, double porcWid, double porcHei) {
        if (porcWid <= 0 && porcHei <= 0)
            return ret;
        return new Rect((int) ret.tl().x + (int) (ret.width * (porcWid <= 0 ? porcWid : (porcWid / 2))),
                (int) (ret.tl().y + ret.height * (porcHei <= 0 ? porcHei : (porcHei / 2))),
                (int) (ret.width * (porcWid <= 0 ? 1 : 1 - (porcWid / 2))),
                (int) (ret.height * (porcHei <= 0 ? 1 : 1 - (porcHei / 2))));
    }


    private Rect redimensionaROI(Rect rect, int cod) {
        Rect retorno;
        switch (cod) {
            case OLHOS:
                retorno = new Rect(new Point(rect.x,
                        rect.y + (int) (rect.height * 0.2)),
                        new Point(rect.x + rect.width,
                                rect.y + (int) (rect.height * 0.6)));
                break;

            case NARIZ:
                retorno = new Rect(
                        new Point(rect.x + (int) (rect.width * 0.2),
                                rect.y + (int) (rect.height * 0.25)),
                        new Point(rect.x + (int) (rect.width * 0.8),
                                rect.y + (int) (rect.height * 0.8)));
                break;

            case BOCA:
                retorno = new Rect(
                        new Point(rect.x + (int) (rect.width * 0.2),
                                rect.y + (int) (rect.height * 0.6)),
                        new Point(rect.x +  (int) (rect.width * 0.8),
                                rect.y + rect.height));
                break;
            default:
                retorno = rect;
        }
        return retorno;
    }

    //>>>>resto de codigo<<<<<
    //for (int i = 0; i < facesArray.length; i++) {
    //  Core.rectangle(mRgba, facesArray[i].tl(),
    //         facesArray[i].br(), new Scalar(100), 3);
    //getSubMatFromRect(facesArray[i], mCascadeER);
               /* texto(mRgba,facesArray[i].tl(),
                        "Ponto("+facesArray[i].tl().x+","+facesArray[i].tl().y+")"
                        ,1.5,new Scalar(255,255));
                texto(mRgba,new Point(facesArray[i].tl().x+facesArray[i].width,facesArray[i].tl().y),
                        "Ponto(2)"
                        ,1.5,new Scalar(255,255));
                texto(mRgba,new Point(facesArray[i].tl().x,facesArray[i].tl().y+facesArray[i].height),
                        "Ponto(3)"
                        ,1.5,new Scalar(255,255));
                texto(mRgba,facesArray[i].br(),
                        "Ponto(4)"
                        ,1.5,new Scalar(255,255));
                        */
    //}
    /*
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
// Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemSwitchCamera = menu.add("Toggle Front / Back camera");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        String toastMesage = "";
        if (item == mItemSwitchCamera) {
            mOpenCvCameraView.setVisibility
                    (SurfaceView.GONE);
            mIsFrontCamera = !mIsFrontCamera;
            mOpenCvCameraView = (CameraBridgeViewBase)
                    findViewById(R.id.java_surface_view);
            if (mIsFrontCamera) {
                mOpenCvCameraView.setCameraIndex(1);
                toastMesage = "Front Camera";
            } else {
                mOpenCvCameraView.setCameraIndex(-1);
                toastMesage = "Back Camera";
            }
            mOpenCvCameraView.setVisibility
                    (SurfaceView.VISIBLE);
            mOpenCvCameraView
                    .setCvCameraViewListener(this);
            mOpenCvCameraView.enableView();
            Toast toast = Toast.makeText(this,
                    toastMesage, Toast.LENGTH_LONG);
            toast.show();
        }
        return true;
    }*/
}
