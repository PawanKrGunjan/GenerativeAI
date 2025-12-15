document.addEventListener('DOMContentLoaded', function() {
    const micBtn = document.getElementById('microphone');
    const userText = document.getElementById('userText');
    const responseText = document.getElementById('responseText');
    const responseAudio = document.getElementById('response-audio');
    const statusDiv = document.getElementById('status');
    
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;

    // Health check interval
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            const data = await res.json();
            const status = data.model_loaded ? '✅ READY' : '⏳ LOADING...';
            statusDiv.textContent = `Model Status: ${status}`;
            statusDiv.className = data.model_loaded ? 'ready' : 'loading';
        } catch(e) {
            statusDiv.textContent = '❌ Server Error';
            statusDiv.className = 'error';
        }
    }
    checkHealth();
    setInterval(checkHealth, 3000);

    // Recording
    micBtn.addEventListener('click', async () => {
        if (isRecording) {
            stopRecording();
            return;
        }

        if (!navigator.mediaDevices?.getUserMedia) {
            alert('Microphone not supported');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1
                } 
            });
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            audioChunks = [];
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = processRecording;
            
            mediaRecorder.start();
            isRecording = true;
            micBtn.classList.add('recording');
            userText.textContent = '🎙️ Recording... (click to stop)';
            
        } catch(err) {
            alert('Microphone access denied');
            console.error(err);
        }
    });

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        if (mediaRecorder?.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        isRecording = false;
        micBtn.classList.remove('recording');
        userText.textContent = 'Processing...';
    }

    async function processRecording() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        userText.textContent = '⏳ Transcribing...';

        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.webm');

        try {
            // Upload audio
            const uploadRes = await fetch('/process_audio', {
                method: 'POST',
                body: formData
            });
            const uploadData = await uploadRes.json();
            const text = uploadData.text;
            
            if (!text || text.includes('No speech')) {
                userText.textContent = '❌ No speech detected';
                return;
            }
            
            userText.textContent = `"${text}"`;
            responseText.textContent = '⏳ Thinking...';

            // Get LLM response
            const llmRes = await fetch('/get_response', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const llmData = await llmRes.json();
            const response_text = llmData.response_text;
            responseText.textContent = `"${response_text}"`;

            // Generate speech
            const ttsRes = await fetch('/synthesize_speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: response_text })
            });
            const ttsData = await ttsRes.json();
            const audio_filename = ttsData.audio_filename;

            // Play audio
            responseAudio.src = `/audio/${audio_filename}`;
            responseAudio.load();
            
            responseAudio.play().catch(err => {
                console.log('Autoplay blocked:', err);
                responseText.textContent += ' (Click play button)';
            });
            
        } catch(err) {
            userText.textContent = '❌ Error occurred';
            console.error('Error:', err);
        }
    }
});
