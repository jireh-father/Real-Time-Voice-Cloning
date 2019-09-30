//navigator.getUserMedia = (navigator.getUserMedia ||
//                            navigator.webkitGetUserMedia ||
//                            navigator.mozGetUserMedia ||
//                            navigator.msGetUserMedia);
navigator.mediaDevices.getUserMedia({ audio: true, sampleRate: 16000 })
  .then(stream => {
    mediaRecorder = new MediaRecorder(stream);

    var audioChunks = [];
    mediaRecorder.addEventListener("dataavailable", event => {
      audioChunks.push(event.data);
    });

    mediaRecorder.addEventListener("stop", () => {
      const audioBlob = new Blob(audioChunks);


      var xhr=new XMLHttpRequest();
          xhr.onload=function(e) {
              if(this.readyState === 4) {
                  console.log("Server returned: ",e.target.responseText);
              }
          };
            var fd=new FormData();
          fd.append("audio_data",audioBlob, "test.webm");
          xhr.open("POST","record",true);
          xhr.send(fd);
    });

    var recordButton = document.getElementById("recordButton");
    var stopButton = document.getElementById("stopButton");

    //add events to those 2 buttons
    recordButton.addEventListener("click", startRecording);
    stopButton.addEventListener("click", stopRecording);

    function startRecording() {
        mediaRecorder.start();
        document.getElementById("recordButton").disabled = true;
    	document.getElementById("stopButton").disabled = false;
    }

    function stopRecording() {
        mediaRecorder.stop();
        document.getElementById("recordButton").disabled = false;
    	document.getElementById("stopButton").disabled = true;
    }

  });

