const DEFAULT_BACKEND_API_URL = 'http://127.0.0.1:8000'

function getBackendApiUrl() {
  const value = import.meta.env.VITE_BACKEND_API_URL ?? import.meta.env.VITE_MODERATION_API_URL
  if (typeof value === 'string' && value.trim().length > 0) {
    return value.trim().replace(/\/+$/, '')
  }
  return DEFAULT_BACKEND_API_URL
}

export async function deletePostViaBackend(postId: string, accessToken: string): Promise<void> {
  const response = await fetch(`${getBackendApiUrl()}/posts/${postId}`, {
    method: 'DELETE',
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(errorText || `Delete request failed with status ${response.status}.`)
  }
}
