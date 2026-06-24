const DEFAULT_MODERATION_API_URL = 'http://127.0.0.1:8000'
const REQUEST_TIMEOUT_MS = 30000

type ModerationApiItem = {
  harmful: boolean
  label: 0 | 1
  ensemble_probability: number
  multimodal_probability: number
  caption_probability: number
}

type ModerationApiResponse = {
  count: number
  multimodal_weight: number
  caption_weight: number
  ensemble_threshold: number
  items: ModerationApiItem[]
}

export type ModerationVerdict = {
  harmful: boolean
  label: 0 | 1
  ensembleProbability: number
  multimodalProbability: number
  captionProbability: number
  ensembleThreshold: number
  multimodalWeight: number
  captionWeight: number
}

function getModerationApiUrl() {
  const value = import.meta.env.VITE_MODERATION_API_URL
  if (typeof value === 'string' && value.trim().length > 0) {
    return value.trim().replace(/\/+$/, '')
  }
  return DEFAULT_MODERATION_API_URL
}

async function fileToBase64(file: File) {
  const bytes = new Uint8Array(await file.arrayBuffer())
  const chunkSize = 0x8000
  let binary = ''

  for (let index = 0; index < bytes.length; index += chunkSize) {
    const chunk = bytes.subarray(index, index + chunkSize)
    binary += String.fromCharCode(...chunk)
  }

  return btoa(binary)
}

export async function classifyPostBeforePublish(
  caption: string,
  imageFile: File,
): Promise<ModerationVerdict> {
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    const response = await fetch(`${getModerationApiUrl()}/predict-batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify([
        {
          caption,
          image_base64: await fileToBase64(imageFile),
        },
      ]),
      signal: controller.signal,
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(errorText || `Moderation request failed with status ${response.status}.`)
    }

    const payload = (await response.json()) as ModerationApiResponse
    const item = payload.items[0]
    if (!item) {
      throw new Error('Moderation API returned an empty result.')
    }

    return {
      harmful: item.harmful,
      label: item.label,
      ensembleProbability: item.ensemble_probability,
      multimodalProbability: item.multimodal_probability,
      captionProbability: item.caption_probability,
      ensembleThreshold: payload.ensemble_threshold,
      multimodalWeight: payload.multimodal_weight,
      captionWeight: payload.caption_weight,
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Moderation request timed out. Make sure the FastAPI service is running.')
    }
    throw error
  } finally {
    window.clearTimeout(timeoutId)
  }
}
