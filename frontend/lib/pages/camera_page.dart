import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? _controller;
  bool _isDetecting = false;
  String? _prediction;
  double _fps = 0;
  int _frameCount = 0;
  late Timer _fpsTimer;

  final String backendUrl = "http://127.0.0.1:8000/detect"; // Change to your IP if mobile

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _startFPSTimer();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final camera = cameras.first;

    _controller = CameraController(camera, ResolutionPreset.medium, enableAudio: false);
    await _controller!.initialize();
    if (!mounted) return;
    setState(() {});

    _controller!.startImageStream((CameraImage image) {
      if (!_isDetecting) {
        _isDetecting = true;
        _processFrame(image);
      }
    });
  }

  void _startFPSTimer() {
    _fpsTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      setState(() {
        _fps = _frameCount.toDouble();
        _frameCount = 0;
      });
    });
  }

  Future<void> _processFrame(CameraImage image) async {
    try {
      // Convert image to temporary file
      final directory = await getTemporaryDirectory();
      final imagePath = path.join(directory.path, '${DateTime.now().millisecondsSinceEpoch}.jpg');
      final file = File(imagePath);
      await file.writeAsBytes(image.planes[0].bytes);

      // Send to backend
      final request = http.MultipartRequest('POST', Uri.parse(backendUrl))
        ..files.add(await http.MultipartFile.fromPath('file', file.path));

      final response = await request.send();
      if (response.statusCode == 200) {
        final result = await response.stream.bytesToString();
        final data = jsonDecode(result);
        setState(() => _prediction = data['detections']?.join(', ') ?? 'No detection');
      } else {
        setState(() => _prediction = 'Error: ${response.statusCode}');
      }

      _frameCount++;
    } catch (e) {
      debugPrint('Error processing frame: $e');
    } finally {
      _isDetecting = false;
    }
  }

  @override
  void dispose() {
    _fpsTimer.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          CameraPreview(_controller!),
          Positioned(
            top: 40,
            left: 20,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildInfoBox("FPS", _fps.toStringAsFixed(1)),
                const SizedBox(height: 8),
                _buildInfoBox("Prediction", _prediction ?? "Waiting..."),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoBox(String label, String value) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        "$label: $value",
        style: const TextStyle(color: Colors.white, fontSize: 16),
      ),
    );
  }
}
