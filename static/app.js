const scoreBtn = document.getElementById('scoreBtn');
const statusEl = document.getElementById('status');
const atsScoreEl = document.getElementById('atsScore');
const matchPercentEl = document.getElementById('matchPercent');
const predictedIndustryEl = document.getElementById('predictedIndustry');
const modelConfidenceEl = document.getElementById('modelConfidence');
const matchedKeywordsEl = document.getElementById('matchedKeywords');
const missingKeywordsEl = document.getElementById('missingKeywords');

const setStatus = (message, type = '') => {
  statusEl.textContent = message;
  statusEl.className = `status ${type}`.trim();
};

const renderKeywordList = (container, list, kind = 'normal') => {
  container.innerHTML = '';

  if (!list.length) {
    container.textContent = 'No keywords found';
    return;
  }

  list.forEach((keyword) => {
    const tag = document.createElement('span');
    tag.className = `keyword ${kind === 'missing' ? 'missing' : ''}`;
    tag.textContent = keyword;
    container.appendChild(tag);
  });
};

scoreBtn.addEventListener('click', async () => {
  const resumeText = document.getElementById('resumeText').value.trim();
  const jdText = document.getElementById('jdText').value.trim();

  if (!resumeText || !jdText) {
    setStatus('Please enter both resume text and job description.', 'error');
    return;
  }

  setStatus('Analyzing resume against the job description...', 'success');
  scoreBtn.disabled = true;

  try {
    const response = await fetch('/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resumeText, jdText })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Something went wrong while scoring the resume.');
    }

    const data = await response.json();

    atsScoreEl.textContent = `${data.ats_score || 0}%`;
    matchPercentEl.textContent = `${data.match_percentage || 0}%`;
    predictedIndustryEl.textContent = data.predicted_industry || 'Unknown';
    modelConfidenceEl.textContent = `${data.model_confidence || 0}%`;

    renderKeywordList(matchedKeywordsEl, data.matched_keywords || [], 'normal');
    renderKeywordList(missingKeywordsEl, data.missing_keywords || [], 'missing');
    setStatus('Resume analysis complete.', 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    scoreBtn.disabled = false;
  }
});
