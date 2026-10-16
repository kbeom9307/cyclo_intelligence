async function request(path, method = 'GET', body) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
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
