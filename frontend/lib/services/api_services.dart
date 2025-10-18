import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://127.0.0.1:8000"; // change for mobile testing (use local IP)

  static Future<Map<String, dynamic>> sendFrame(File imageFile) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse("$baseUrl/predict/static/"),
    );
    request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));
    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Failed to get prediction");
    }
  }
}
