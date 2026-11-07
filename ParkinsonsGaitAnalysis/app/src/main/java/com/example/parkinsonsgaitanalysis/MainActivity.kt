
package com.example.parkinsonsgaitanalysis

import android.content.ContentValues
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.parkinsonsgaitanalysis.ui.theme.ParkinsonsGaitAnalysisTheme
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null

    private val readings = mutableListOf<String>()

    var isRecording by mutableStateOf(false)
        private set

    var participantId by mutableStateOf("")
        private set

    var liveX by mutableFloatStateOf(0f)
        private set

    var liveY by mutableFloatStateOf(0f)
        private set

    var liveZ by mutableFloatStateOf(0f)
        private set

    var readingCount by mutableIntStateOf(0)
        private set

    var statusMessage by mutableStateOf("Ready to start")
        private set

    var savedFile by mutableStateOf("")
        private set

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager =
            getSystemService(SENSOR_SERVICE) as SensorManager

        accelerometer =
            sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        setContent {
            ParkinsonsGaitAnalysisTheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize()
                ) { innerPadding ->
                    MainScreen(
                        modifier = Modifier.padding(innerPadding),
                        participantId = participantId,
                        isRecording = isRecording,
                        liveX = liveX,
                        liveY = liveY,
                        liveZ = liveZ,
                        readingCount = readingCount,
                        statusMessage = statusMessage,
                        savedFile = savedFile,
                        onStart = { name ->
                            startRecording(name)
                        },
                        onStop = {
                            stopRecording()
                        }
                    )
                }
            }
        }
    }

    private fun startRecording(name: String) {

        if (name.isBlank()) {
            Toast.makeText(
                this,
                "Please enter participant name",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        val sensor = accelerometer

        if (sensor == null) {
            statusMessage = "Accelerometer not available"
            return
        }

        readings.clear()
        readingCount = 0
        savedFile = ""

        participantId =
            "P-" + UUID.randomUUID()
                .toString()
                .take(8)
                .uppercase(Locale.US)

        val registered = sensorManager.registerListener(
            this,
            sensor,
            SensorManager.SENSOR_DELAY_GAME
        )

        if (!registered) {
            statusMessage = "Could not start sensor"
            return
        }

        isRecording = true
        statusMessage = "Recording in progress..."
        liveX = 0f
        liveY = 0f
        liveZ = 0f
    }

    override fun onSensorChanged(event: SensorEvent?) {

        if (!isRecording || event == null) return

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {

            val x = event.values[0]
            val y = event.values[1]
            val z = event.values[2]

            val timestamp = SimpleDateFormat(
                "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'",
                Locale.US
            ).apply {
                timeZone = TimeZone.getTimeZone("UTC")
            }.format(Date(System.currentTimeMillis()))

            val row = String.format(
                Locale.US,
                "%s,%d,%.6f,%.6f,%.6f",
                timestamp,
                event.timestamp,
                x,
                y,
                z
            )

            readings.add(row)
            readingCount++

            // Update the screen every five readings.
            if (readingCount % 5 == 0) {
                liveX = x
                liveY = y
                liveZ = z
            }
        }
    }

    override fun onAccuracyChanged(
        sensor: Sensor?,
        accuracy: Int
    ) {
        // No action required.
    }

    private fun stopRecording() {

        if (!isRecording) return

        sensorManager.unregisterListener(this)
        isRecording = false

        statusMessage = "Saving recorded data..."

        if (readings.isEmpty()) {
            statusMessage = "No sensor readings collected"
            return
        }

        try {
            val fileName = saveCsv()

            savedFile = fileName
            statusMessage = "Recording saved successfully!"

            Toast.makeText(
                this,
                "CSV saved successfully",
                Toast.LENGTH_LONG
            ).show()

        } catch (e: Exception) {
            statusMessage = "Error saving CSV: ${e.message}"

            Toast.makeText(
                this,
                statusMessage,
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun saveCsv(): String {

        val fileName = "gait_${participantId}_" +
                SimpleDateFormat(
                    "yyyyMMdd_HHmmss",
                    Locale.US
                ).format(Date()) + ".csv"

        val csvContent = buildString {
            append("participant_id,timestamp_utc,elapsed_realtime_ns,x_m_s2,y_m_s2,z_m_s2\n")
            readings.forEach { row ->
                append(participantId)
                append(",")
                append(row)
                append("\n")
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {

            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, "text/csv")
                put(
                    MediaStore.MediaColumns.RELATIVE_PATH,
                    Environment.DIRECTORY_DOWNLOADS +
                            "/ParkinsonsGaitAnalysis"
                )
            }

            val uri: Uri = contentResolver.insert(
                MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                values
            ) ?: throw Exception("Could not create CSV file")

            try {
                contentResolver.openOutputStream(uri)?.bufferedWriter()
                    ?.use { writer ->
                        writer.write(csvContent)
                    } ?: throw Exception("Could not write CSV file")
            } catch (e: Exception) {
                contentResolver.delete(uri, null, null)
                throw e
            }

            return "Downloads/ParkinsonsGaitAnalysis/$fileName"

        } else {

            val directory = getExternalFilesDir(
                Environment.DIRECTORY_DOCUMENTS
            ) ?: filesDir

            if (!directory.exists()) {
                directory.mkdirs()
            }

            val file = File(directory, fileName)
            file.writeText(csvContent)

            return file.absolutePath
        }
    }

    override fun onDestroy() {
        sensorManager.unregisterListener(this)
        super.onDestroy()
    }
}

@Composable
fun MainScreen(
    modifier: Modifier = Modifier,
    participantId: String,
    isRecording: Boolean,
    liveX: Float,
    liveY: Float,
    liveZ: Float,
    readingCount: Int,
    statusMessage: String,
    savedFile: String,
    onStart: (String) -> Unit,
    onStop: () -> Unit
) {

    var participantName by remember {
        mutableStateOf("")
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {

        Text(
            text = "Parkinson's Gait Analysis",
            fontSize = 26.sp
        )

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = "Smartphone-Based Gait Assessment",
            fontSize = 16.sp
        )

        Spacer(modifier = Modifier.height(28.dp))

        OutlinedTextField(
            value = participantName,
            onValueChange = { participantName = it },
            label = { Text("Participant Name") },
            placeholder = { Text("Enter participant name") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            enabled = !isRecording
        )

        Spacer(modifier = Modifier.height(16.dp))

        if (participantId.isNotEmpty()) {
            Text(
                text = "Participant ID: $participantId",
                fontSize = 16.sp
            )
        }

        Spacer(modifier = Modifier.height(20.dp))

        Button(
            onClick = {
                onStart(participantName.trim())
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isRecording && participantName.isNotBlank()
        ) {
            Text("Start Walking Test")
        }

        Spacer(modifier = Modifier.height(12.dp))

        OutlinedButton(
            onClick = onStop,
            modifier = Modifier.fillMaxWidth(),
            enabled = isRecording
        ) {
            Text("Stop Recording")
        }

        Spacer(modifier = Modifier.height(24.dp))

        HorizontalDivider()

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Recording Status",
            fontSize = 20.sp
        )

        Spacer(modifier = Modifier.height(12.dp))

        Text(text = statusMessage)

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = "Readings Collected: $readingCount"
        )

        Spacer(modifier = Modifier.height(20.dp))

        Text(
            text = "Live Accelerometer Data",
            fontSize = 18.sp
        )

        Spacer(modifier = Modifier.height(12.dp))

        Text(text = "X-axis: %.4f m/s²".format(liveX))
        Text(text = "Y-axis: %.4f m/s²".format(liveY))
        Text(text = "Z-axis: %.4f m/s²".format(liveZ))

        Spacer(modifier = Modifier.height(20.dp))

        if (savedFile.isNotEmpty()) {
            HorizontalDivider()
            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = "CSV File Saved",
                fontSize = 18.sp
            )

            Text(
                text = savedFile,
                fontSize = 12.sp
            )
        }
    }
}
