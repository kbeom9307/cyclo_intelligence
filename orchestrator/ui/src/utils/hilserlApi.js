async function request(path, method = 'GET', body, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`/api/hilserl${path}`, {
      method, signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail;
      throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
        ? detail.map((item) => `${item.loc?.slice(1).join('.')}: ${item.msg}`).join('; ')
        : `Request failed (${response.status})`);
    }
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('Request timed out. Refresh to check the latest session state.');
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
export const getHilWorkspace = () => request('/workspace');
export const saveHilConfig = (config, revision) => request('/config', 'PUT', { config, expected_revision: revision });
export const createHilSession = (revision, mode = 'preview') => request('/sessions', 'POST', { expected_config_revision: revision, mode });
export const sendHilCommand = (session, action) => request(`/sessions/${encodeURIComponent(session.id)}/commands`, 'POST', { action, expected_revision: session.revision });

export const sendHilAction = (session, value) => request(`/sessions/${encodeURIComponent(session.id)}/action`, 'POST', { value, expected_revision: session.revision });
export const trainHilClassifier = (session_ids, epochs = 20, threshold = .9) => request('/classifiers/train', 'POST', { session_ids, epochs, threshold, seed: 0 });
export const sendHilClassifierLabel = (session, success) => request(`/sessions/${encodeURIComponent(session.id)}/classifier-label`, 'POST', { success, expected_revision: session.revision });

export const deleteHilDataset = (id, kind) => request(`/datasets/${encodeURIComponent(id)}/${kind}`, 'DELETE');
export const deleteHilClassifier = id => request(`/classifiers/${encodeURIComponent(id)}`, 'DELETE');
export const stopHilClassifier = id => request(`/classifiers/${encodeURIComponent(id)}/stop`, 'POST');
export const deleteHilDemoEpisodes = (id, episodes) => request(`/datasets/${encodeURIComponent(id)}/demo/delete-episodes`, 'POST', { episodes });
export const combineHilDemos = selections => request('/datasets/demo/combine', 'POST', { selections }, 120000);

export const checkHilLearner = config => request('/learner/check', 'POST', config);

export const restartHilTraining = session => request(`/sessions/${encodeURIComponent(session.id)}/restart-training`, 'POST', {expected_revision: session.revision});
