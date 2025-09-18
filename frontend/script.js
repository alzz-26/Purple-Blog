document.getElementById('predictBtn').onclick = async () => {
  const content = document.getElementById('content').value;
  const publishTime = document.getElementById('publishTime').value;
  const res = await fetch('http://127.0.0.1:5000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, publishTime })
  });
  const data = await res.json();
  document.getElementById('result').innerHTML = `
    Predicted Likes: ${data.predictedLikes}<br>
    Predicted Comments: ${data.predictedComments}
  `;
};
