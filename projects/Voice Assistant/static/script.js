let lightMode = true;
let recorder = null;
let recording = false;
let voiceOption = "default";
const responses = [];
const botRepeatButtonIDToIndexMap = {};
const userRepeatButtonIDToRecordingMap = {};
const baseUrl = window.location.origin;
let mediaRecorder;
let audioChunks = [];

async function showBotLoadingAnimation() {
  await sleep(500);
  $(".loading-animation")[1].style.display = "inline-block";
}

function hideBotLoadingAnimation() {
  $(".loading-animation")[1].style.display = "none";
}

async function showUserLoadingAnimation() {
  await sleep(100);
  $(".loading-animation")[0].style.display = "flex";
}

function hideUserLoadingAnimation() {
  $(".loading-animation")[0].style.display = "none";
}

const getSpeechToText = async (audioBlob) => {
  const formData = new FormData();
  formData.append('audio_data', audioBlob, 'recording.wav');
  
  let response = await $.ajax({
    url: baseUrl + "/speech-to-text",
    type: "POST",
    data: formData,
    processData: false,
    contentType: false
  });
  
  return response.text;
};

const processUserMessage = async (userMessage) => {
  let response = await fetch(baseUrl + "/process-message", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ userMessage: userMessage, voice: voiceOption }),
  });
  response = await response.json();
  return response;
};

const cleanTextInput = (value) => {
  return value
    .trim() // remove starting and ending spaces
    .replace(/[\n\t]/g, "") // remove newlines and tabs
    .replace(/<[^>]*>/g, "") // remove HTML tags
    .replace(/[<>&;]/g, ""); // sanitize inputs
};

function startRecording() {
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    audioChunks = [];
    mediaRecorder.ondataavailable = event => {
      audioChunks.push(event.data);
    };
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      processRecordedAudio(audioBlob);
    };
    recording = true;
    document.getElementById("micButton").textContent = "🎤 Stop Recording";
  });
}

function stopRecording() {
  mediaRecorder.stop();
  recording = false;
  document.getElementById("micButton").textContent = "🎤 Start Recording";
}

function processRecordedAudio(audioBlob) {
  showUserLoadingAnimation();

  // Call getSpeechToText and print the result
  getSpeechToText(audioBlob).then(userMessage => {
    // Print the result to the console
    console.log("Transcribed Text:", userMessage);

    // Continue with the rest of the logic
    populateUserMessage(userMessage, audioBlob);
    populateBotResponse(userMessage);
  }).catch(error => {
    console.error("Error in getSpeechToText:", error);
  });
}

const toggleRecording = () => {
  if (!recording) {
    startRecording();
  } else {
    stopRecording();
  }
};

const sleep = (time) => new Promise((resolve) => setTimeout(resolve, time));

const populateUserMessage = (userMessage, audioBlob) => {
  $("#message-input").val(""); // Clear the input field

  if (audioBlob) {
    const userRepeatButtonID = getRandomID();
    userRepeatButtonIDToRecordingMap[userRepeatButtonID] = audioBlob;
    hideUserLoadingAnimation();
    $("#message-list").append(
      `<div class='message-line my-text'><div class='message-box my-text${
        !lightMode ? " dark" : ""
      }'><div class='me'>${userMessage}</div></div>
            <button id='${userRepeatButtonID}' class='btn volume repeat-button' onclick='userRepeatButtonIDToRecordingMap[this.id].play()'><i class='fa fa-volume-up'></i></button>
            </div>`
    );
  } else {
    $("#message-list").append(
      `<div class='message-line my-text'><div class='message-box my-text${
        !lightMode ? " dark" : ""
      }'><div class='me'>${userMessage}</div></div></div>`
    );
  }

  scrollToBottom();
};

const populateBotResponse = async (userMessage) => {
  await showBotLoadingAnimation();
  const response = await processUserMessage(userMessage);
  responses.push(response);

  const repeatButtonID = getRandomID();
  botRepeatButtonIDToIndexMap[repeatButtonID] = responses.length - 1;
  hideBotLoadingAnimation();
  // Append the random message to the message list
  $("#message-list").append(
    `<div class='message-line'><div class='message-box${
      !lightMode ? " dark" : ""
    }'>${
      response.openaiResponseText
    }</div><button id='${repeatButtonID}' class='btn volume repeat-button' onclick='playResponseAudio("data:audio/wav;base64," + responses[botRepeatButtonIDToIndexMap[this.id]].openaiResponseSpeech);console.log(this.id)'><i class='fa fa-volume-up'></i></button></div>`
  );

  playResponseAudio("data:audio/wav;base64," + response.openaiResponseSpeech);

  scrollToBottom();
};

const playResponseAudio = (function () {
  const df = document.createDocumentFragment();
  return function Sound(src) {
    const snd = new Audio(src);
    df.appendChild(snd); // keep in fragment until finished playing
    snd.addEventListener("ended", function () {
      df.removeChild(snd);
    });
    snd.play();
    return snd;
  };
})();

const getRandomID = () => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
};

const scrollToBottom = () => {
  // Scroll the chat window to the bottom
  $("#chat-window").animate({
    scrollTop: $("#chat-window")[0].scrollHeight,
  });
};

$(document).ready(function () {
  $("#message-input").keyup(function (event) {
    let inputVal = cleanTextInput($("#message-input").val());

    if (event.keyCode === 13 && inputVal != "") {
      const message = inputVal;

      populateUserMessage(message, null);
      populateBotResponse(message);
    }

    inputVal = $("#message-input").val();

    if (inputVal == "" || inputVal == null) {
      $("#send-button")
        .removeClass("send")
        .addClass("microphone")
        .html("<i class='fa fa-microphone'></i>");
    } else {
      $("#send-button")
        .removeClass("microphone")
        .addClass("send")
        .html("<i class='fa fa-paper-plane'></i>");
    }
  });

  // When the user clicks the "Send" button
  $("#send-button").click(function () {
    if ($("#send-button").hasClass("microphone")) {
      toggleRecording();
      $(".fa-microphone").css("color", recording ? "#f44336" : "#125ee5");
    } else {
      const message = cleanTextInput($("#message-input").val());
      populateUserMessage(message, null);
      populateBotResponse(message);
      $("#send-button")
        .removeClass("send")
        .addClass("microphone")
        .html("<i class='fa fa-microphone'></i>");
    }
  });

  $("#light-dark-mode-switch").change(function () {
    $("body").toggleClass("dark-mode");
    $(".message-box").toggleClass("dark");
    $(".loading-dots").toggleClass("dark");
    $(".dot").toggleClass("dark-dot");
    lightMode = !lightMode;
  });

  $("#voice-options").change(function () {
    voiceOption = $(this).val();
  });
});
