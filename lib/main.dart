
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart' show YOLOViewController;



void main() {
    runApp(MyApp());
}

class MyApp extends StatelessWidget {
    @override
    Widget build(BuildContext context) {
        return MaterialApp(
            title: 'YOLO Face Detection',
            home: YOLOClassificationPage(),
            debugShowCheckedModeBanner: false,
            theme: ThemeData(
                primarySwatch: Colors.blue,
                visualDensity: VisualDensity.adaptivePlatformDensity,
            ),
        );
    }
}

class YOLOClassificationPage extends StatefulWidget {
    @override
    _YOLOClassificationPageState createState() => _YOLOClassificationPageState();
}

class _YOLOClassificationPageState extends State<YOLOClassificationPage> {
    YOLOViewController? controller;
    List<YOLOResult> results = [];
    bool hasPermission = false;
    bool isLoading = true;
    bool isFrontCamera = true;


    final String backCameraModel =
        'best_float16(revfix).tflite';
    final String frontCameraModel =
        'best_float16(revfix).tflite';


    double confidenceThreshold = 0.7;
    double iouThreshold = 0.5;
    int numItemsThreshold = 1;


    String? lastDetectedClass;
    int sameClassCount = 0;
    final int minSameClassCount = 2;


    double currentFPS = 0.0;
    double processingTime = 0.0;
    int frameCount = 0;


    bool debugMode = false;

    @override
    void initState() {
        super.initState();
        controller = YOLOViewController();
        _requestPermission();
        _initializeController();


        WidgetsBinding.instance.addPostFrameCallback((_) {
            Future.delayed(Duration(milliseconds: 500), () {
                _switchToFrontCamera();
            });
        });
    }

    Future<void> _initializeController() async {
        await Future.delayed(Duration(milliseconds: 500));
        try {
            await controller?.setThresholds(
                confidenceThreshold: confidenceThreshold,
                iouThreshold: iouThreshold,
                numItemsThreshold: numItemsThreshold,
            );
            print('Controller initialized with strict thresholds');
        } catch (e) {
            print('Error setting thresholds: $e');
        }
    }

    Future<void> _requestPermission() async {
        final status = await Permission.camera.request();
        setState(() {
            hasPermission = status.isGranted;
            isLoading = false;
        });
    }

    Future<void> _switchToFrontCamera() async {
        try {

            await controller?.switchCamera();


            await controller?.switchModel(frontCameraModel, YOLOTask.detect);

            setState(() {
                isFrontCamera = true;
                lastDetectedClass = null;
                sameClassCount = 0;
            });

            print('Switched to front camera with flipped model: $frontCameraModel');
        } catch (e) {
            print('Error switching to front camera: $e');
        }
    }

    void _onResult(List<YOLOResult> detectionResults) {
        frameCount++;

        if (detectionResults.isEmpty) {
            setState(() {
                results = [];
            });
            return;
        }


        List<YOLOResult> filteredResults = detectionResults
            .where((result) => result.confidence >= confidenceThreshold)
            .toList();


        filteredResults.sort((a, b) => b.confidence.compareTo(a.confidence));


        List<YOLOResult> topResults = filteredResults.take(3).toList();


        if (topResults.isNotEmpty) {
            String currentTopClass = topResults.first.className;

            if (currentTopClass == lastDetectedClass) {
                sameClassCount++;
            } else {
                lastDetectedClass = currentTopClass;
                sameClassCount = 1;
            }


            if (sameClassCount >= minSameClassCount) {
                setState(() {
                    results = topResults;
                });
            }
        } else {
            setState(() {
                results = [];
            });
        }
    }


    Future<void> _switchCamera() async {
        try {

            String modelToUse = isFrontCamera ? backCameraModel : frontCameraModel;


            await controller?.switchCamera();


            await controller?.switchModel(modelToUse, YOLOTask.detect);

            setState(() {
                isFrontCamera = !isFrontCamera;
                lastDetectedClass = null;
                sameClassCount = 0;
                frameCount = 0;
            });

            print(
                'Switched to ${isFrontCamera ? "front" : "back"} camera with model: $modelToUse',
            );
        } catch (e) {
            print('Error switching camera: $e');
        }
    }

    String _getTopResult() {
        if (results.isEmpty) return 'Scanning...';
        return results.first.className;
    }

    double _getTopConfidence() {
        if (results.isEmpty) return 0.0;
        return results.first.confidence;
    }

    Color _getConfidenceColor(double confidence) {
        if (confidence > 0.9) return Colors.green;
        if (confidence > 0.8) return Colors.lightGreen;
        if (confidence > 0.7) return Colors.yellow[700]!;
        if (confidence > 0.6) return Colors.orange;
        return Colors.red;
    }

    String _getConfidenceEmoji(double confidence) {
        if (confidence > 0.9) return 'Sangat Baik';
        if (confidence > 0.8) return 'Baik';
        if (confidence > 0.7) return 'Oke';
        if (confidence > 0.6) return 'Ragu';
        return '❓';
    }


    bool _isFacingFront() {
        if (results.isEmpty) return true;
        String topResult = _getTopResult().toLowerCase();

        return topResult.contains('front') ||
            topResult.contains('depan') ||
            topResult == 'scanning...';
    }


    bool _isNoFaceDetected() {
        return results.isEmpty || _getTopConfidence() < 0.7;
    }

    @override
    Widget build(BuildContext context) {
        if (isLoading) {
            return Scaffold(
                backgroundColor: Colors.black,
                body: Center(
                    child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                            CircularProgressIndicator(color: Colors.white),
                            SizedBox(height: 16),
                            Text('Memuat...', style: TextStyle(color: Colors.white)),
                        ],
                    ),
                ),
            );
        }

        if (!hasPermission) {
            return Scaffold(
                backgroundColor: Colors.grey[900],
                body: Center(
                    child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                            Icon(Icons.camera_alt, size: 80, color: Colors.white70),
                            SizedBox(height: 24),
                            Text(
                                'Akses Kamera Diperlukan',
                                style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 20,
                                    fontWeight: FontWeight.bold,
                                ),
                            ),
                            SizedBox(height: 24),
                            ElevatedButton.icon(
                                onPressed: _requestPermission,
                                icon: Icon(Icons.camera),
                                label: Text('Berikan Akses Kamera'),
                                style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.blue,
                                    foregroundColor: Colors.white,
                                    padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                                ),
                            ),
                        ],
                    ),
                ),
            );
        }


        bool isNoFaceDetected = _isNoFaceDetected();
        bool isFacingFront = _isFacingFront();

        return Scaffold(
            backgroundColor: Colors.black,
            body: Stack(
                children: [

                    YOLOView(
                        modelPath: isFrontCamera
                            ? frontCameraModel
                            : backCameraModel,
                        task: YOLOTask.detect,
                        controller: controller,
                        onResult: _onResult,
                        showNativeUI: false,
                        showOverlays: false,
                        useGpu: true,
                        cameraResolution: "720`",
                        onPerformanceMetrics: (metrics) {
                            setState(() {
                                currentFPS = metrics.fps;
                                processingTime = metrics.processingTimeMs;
                            });
                        },
                    ),


                    Center(
                        child: Container(
                            width: 240,
                            height: 320,
                            decoration: BoxDecoration(
                                border: Border.all(
                                    color: results.isNotEmpty && _getTopConfidence() > 0.7
                                        ? _getConfidenceColor(_getTopConfidence())
                                        : Colors.white.withOpacity(0.3),
                                    width: 2,
                                ),
                                borderRadius: BorderRadius.circular(120),
                            ),
                        ),
                    ),


                    SafeArea(
                        child: Container(
                            margin: EdgeInsets.all(16),
                            padding: EdgeInsets.all(16),
                            decoration: BoxDecoration(
                                color: Colors.black.withOpacity(0.8),
                                borderRadius: BorderRadius.circular(12),
                            ),
                            child: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [

                                    Row(
                                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                        children: [
                                            Text(
                                                'Mode: Classification',
                                                style: TextStyle(color: Colors.white70, fontSize: 12),
                                            ),
                                            Text(
                                                '${currentFPS.toStringAsFixed(0)} FPS',
                                                style: TextStyle(
                                                    color: currentFPS > 20 ? Colors.green : Colors.orange,
                                                    fontSize: 12,
                                                ),
                                            ),
                                        ],
                                    ),
                                    SizedBox(height: 12),


                                    if (results.isNotEmpty && _getTopConfidence() > 0.7)
                                        Column(
                                            children: [
                                                Row(
                                                    mainAxisAlignment: MainAxisAlignment.center,
                                                    children: [
                                                        Text(
                                                            _getConfidenceEmoji(_getTopConfidence()),
                                                            style: TextStyle(fontSize: 24),
                                                        ),
                                                        SizedBox(width: 8),
                                                        Text(
                                                            _getTopResult(),
                                                            style: TextStyle(
                                                                color: Colors.white,
                                                                fontSize: 22,
                                                                fontWeight: FontWeight.bold,
                                                            ),
                                                        ),
                                                    ],
                                                ),
                                                SizedBox(height: 8),
                                                Container(
                                                    width: double.infinity,
                                                    height: 6,
                                                    decoration: BoxDecoration(
                                                        color: Colors.white.withOpacity(0.2),
                                                        borderRadius: BorderRadius.circular(3),
                                                    ),
                                                    child: FractionallySizedBox(
                                                        alignment: Alignment.centerLeft,
                                                        widthFactor: _getTopConfidence(),
                                                        child: Container(
                                                            decoration: BoxDecoration(
                                                                color: _getConfidenceColor(_getTopConfidence()),
                                                                borderRadius: BorderRadius.circular(3),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                                SizedBox(height: 4),
                                                Text(
                                                    '${(_getTopConfidence() * 100).toStringAsFixed(1)}% Confidence',
                                                    style: TextStyle(
                                                        color: _getConfidenceColor(_getTopConfidence()),
                                                        fontSize: 13,
                                                    ),
                                                ),
                                            ],
                                        )
                                    else
                                        Text(
                                            'Arahkan wajah ke kamera',
                                            style: TextStyle(color: Colors.white70, fontSize: 16),
                                        ),
                                ],
                            ),
                        ),
                    ),

                    // Camera Switch
                    Positioned(
                        top: 50,
                        right: 16,
                        child: SafeArea(
                            child: Column(
                                children: [
                                    IconButton(
                                        onPressed: _switchCamera,
                                        icon: Container(
                                            padding: EdgeInsets.all(8),
                                            decoration: BoxDecoration(
                                                color: Colors.black.withOpacity(0.6),
                                                shape: BoxShape.circle,
                                            ),
                                            child: Icon(
                                                Icons.cameraswitch,
                                                color: Colors.white,
                                                size: 20,
                                            ),
                                        ),
                                    ),
                                    Text(
                                        isFrontCamera ? 'Front' : 'Back',
                                        style: TextStyle(color: Colors.white70, fontSize: 10),
                                    ),
                                ],
                            ),
                        ),
                    ),


                    Positioned(
                        bottom: 100,
                        left: 16,
                        right: 16,
                        child: Container(
                            padding: EdgeInsets.all(16),
                            decoration: BoxDecoration(
                                color: isNoFaceDetected
                                    ? Colors.purple.withOpacity(0.9)
                                    : isFacingFront
                                    ? Colors.green.withOpacity(0.9)
                                    : Colors.red.withOpacity(0.9),
                                borderRadius: BorderRadius.circular(12),
                                boxShadow: [
                                    BoxShadow(
                                        color: isNoFaceDetected
                                            ? Colors.purple.withOpacity(0.5)
                                            : isFacingFront
                                            ? Colors.green.withOpacity(0.5)
                                            : Colors.red.withOpacity(0.5),
                                        blurRadius: 10,
                                        spreadRadius: 2,
                                    ),
                                ],
                            ),
                            child: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                    Text(
                                        isNoFaceDetected
                                            ? '👤'
                                            : isFacingFront
                                            ? '😁'
                                            : '⚠️',
                                        style: TextStyle(fontSize: 32),
                                    ),
                                    SizedBox(height: 8),
                                    Text(
                                        isNoFaceDetected
                                            ? 'Wajah Tidak Terdeteksi'
                                            : isFacingFront
                                            ? 'Semangat Berbuat Jujur!'
                                            : 'Terdeteksi Mencontek!',
                                        textAlign: TextAlign.center,
                                        style: TextStyle(
                                            color: Colors.white,
                                            fontSize: 18,
                                            fontWeight: FontWeight.bold,
                                        ),
                                    ),
                                ],
                            ),
                        ),
                    ),

                    // Debug Info
                    if (debugMode)
                        Positioned(
                            bottom: 20,
                            left: 16,
                            right: 16,
                            child: Container(
                                padding: EdgeInsets.all(12),
                                decoration: BoxDecoration(
                                    color: Colors.black.withOpacity(0.9),
                                    borderRadius: BorderRadius.circular(8),
                                    border: Border.all(color: Colors.red, width: 1),
                                ),
                                child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                        Text(
                                            'DEBUG INFO',
                                            style: TextStyle(
                                                color: Colors.red,
                                                fontSize: 10,
                                                fontWeight: FontWeight.bold,
                                            ),
                                        ),
                                        SizedBox(height: 4),
                                        Text(
                                            'Model: ${isFrontCamera ? frontCameraModel : backCameraModel}',
                                            style: TextStyle(color: Colors.white70, fontSize: 10),
                                        ),
                                        Text(
                                            'Frames: $frameCount | Stable: ${sameClassCount >= minSameClassCount ? "Yes" : "No"}',
                                            style: TextStyle(color: Colors.white70, fontSize: 10),
                                        ),
                                        Text(
                                            'Processing: ${processingTime.toStringAsFixed(0)}ms',
                                            style: TextStyle(color: Colors.white70, fontSize: 10),
                                        ),
                                        if (results.isNotEmpty)
                                            ...results
                                                .take(3)
                                                .map(
                                                    (r) => Text(
                                                    '${r.className}: ${(r.confidence * 100).toStringAsFixed(1)}%',
                                                    style: TextStyle(
                                                        color: Colors.white60,
                                                        fontSize: 10,
                                                    ),
                                                ),
                                            ),
                                    ],
                                ),
                            ),
                        ),

                    // Debug Toggle
                    Positioned(
                        bottom: 20,
                        right: 16,
                        child: GestureDetector(
                            onTap: () => setState(() => debugMode = !debugMode),
                            child: Container(
                                padding: EdgeInsets.all(8),
                                decoration: BoxDecoration(
                                    color: debugMode ? Colors.red : Colors.black54,
                                    shape: BoxShape.circle,
                                ),
                                child: Icon(Icons.bug_report, color: Colors.white, size: 16),
                            ),
                        ),
                    ),
                ],
            ),
        );
    }

    @override
    void dispose() {
        super.dispose();
    }
}