export const SERVER_URL = 'http://192.168.1.10:8000';

export async function apiGet(path: string) { 
  const response = await fetch(`${SERVER_URL}${path}`);

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}
