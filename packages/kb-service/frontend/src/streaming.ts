/**
 * streamSSE — POST JSON to a server-sent-events endpoint and dispatch events.
 *
 * Auth: stream endpoints accept ?token= query param only (no Authorization header).
 * The caller is responsible for appending the JWT to the URL:
 *   `${path}?token=${encodeURIComponent(token)}`
 *
 * @param url    Full URL including any ?token= param.
 * @param body   Request body — serialised as compact JSON.
 * @param onEvent  Called for every complete SSE frame with (eventType, parsedData).
 * @returns  Resolves after the final chunk; rejects if the reader throws mid-stream
 *           (callers must not rely on stream_end alone to restore UI state).
 */
export async function streamSSE(
  url: string,
  body: unknown,
  onEvent: (event: string, data: Record<string, unknown>) => void,
): Promise<void> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = (await res.json()) as { detail?: string }
      if (data.detail) detail = data.detail
    } catch {
      // fall back to statusText
    }
    throw new Error(`${res.status}: ${detail}`)
  }

  if (!res.body) {
    throw new Error('No response body')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // Split on SSE frame separator.  The last element may be a partial frame.
      const frames = buffer.split('\n\n')
      buffer = frames.pop() ?? ''

      for (const frame of frames) {
        if (!frame.trim()) continue

        let eventType = 'message'
        let dataLine = ''

        for (const line of frame.split('\n')) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            dataLine = line.slice(6)
          }
        }

        if (dataLine) {
          try {
            const data = JSON.parse(dataLine) as Record<string, unknown>
            onEvent(eventType, data)
          } catch {
            // skip malformed data lines
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
