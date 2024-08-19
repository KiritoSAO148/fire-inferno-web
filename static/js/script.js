document.getElementById('startButton').addEventListener('click', function () {
    const videoContainer = document.getElementById('videoContainer');
    const videoStream = document.getElementById('videoStream');
    const stopButton = document.getElementById('stopButton');

    videoStream.src = '/video_feed';
    videoContainer.style.display = 'block';
    stopButton.style.display = 'inline-block';
});

document.getElementById('stopButton').addEventListener('click', function () {
    const videoContainer = document.getElementById('videoContainer');
    const videoStream = document.getElementById('videoStream');
    const stopButton = document.getElementById('stopButton');

    videoStream.src = '';
    videoContainer.style.display = 'none';
    stopButton.style.display = 'none';
});
