import { describe, it, expect } from 'vitest'
import { streamSSE } from '../streaming'

/**
 * Build a ReadableStream from an array of Uint8Array chunks.
 * The stream yields chunks in order, then closes.
 */
function makeStream(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  let idx = 0
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (idx < chunks.length) {
        controller.enqueue(chunks[idx++])
      } else {
        controller.close()
      }
    },
  })
}

/**
 * Build a ReadableStream that yields the given chunks then throws an error.
 */
function makeErrorStream(chunks: Uint8Array[], errorMsg: string): ReadableStream<Uint8Array> {
  let idx = 0
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (idx < chunks.length) {
        controller.enqueue(chunks[idx++])
      } else {
        controller.error(new Error(errorMsg))
      }
    },
  })
}

const enc = new TextEncoder()

function mockFetchOk(stream: ReadableStream<Uint8Array>) {
  return {
    ok: true,
    body: stream,
  } as unknown as Response
}

function mockFetchError(status: number, statusText: string) {
  return {
    ok: false,
    status,
    statusText,
    json: async () => ({ detail: statusText }),
  } as unknown as Response
}

describe('streamSSE', () => {
  it('fires events in order and handles chunk-straddling frames', async () => {
    // Frame 1: 'event: classified\ndata: {"mode":"explore"}\n\n'
    // Frame 2: 'event: status\ndata: {"message":"hello"}\n\n'
    // Split so that frame 1's trailing '\n\n' is in the second chunk.

    const frame1a = enc.encode('event: classified\ndata: {"mode":"explore"}')
    const frame1b = enc.encode('\n\nevent: status\ndata: {"message":"hello"}\n\n')

    const stream = makeStream([frame1a, frame1b])
    global.fetch = async () => mockFetchOk(stream)

    const received: Array<{ event: string; data: Record<string, unknown> }> = []
    await streamSSE('http://test/stream', {}, (event, data) => {
      received.push({ event, data })
    })

    expect(received).toHaveLength(2)
    expect(received[0]).toEqual({ event: 'classified', data: { mode: 'explore' } })
    expect(received[1]).toEqual({ event: 'status', data: { message: 'hello' } })
  })

  it('delivers prior events then rejects when reader throws mid-stream', async () => {
    // First chunk: one complete frame
    // Then the stream errors
    const chunk1 = enc.encode('event: classified\ndata: {"mode":"summarize"}\n\n')

    const stream = makeErrorStream([chunk1], 'network drop')
    global.fetch = async () => mockFetchOk(stream)

    const received: Array<{ event: string; data: Record<string, unknown> }> = []

    await expect(
      streamSSE('http://test/stream', {}, (event, data) => {
        received.push({ event, data })
      }),
    ).rejects.toThrow('network drop')

    // The earlier frame's onEvent was still delivered
    expect(received).toHaveLength(1)
    expect(received[0]).toEqual({ event: 'classified', data: { mode: 'summarize' } })
  })

  it('throws before reading stream when response is not ok', async () => {
    global.fetch = async () => mockFetchError(401, 'Unauthorized')

    await expect(
      streamSSE('http://test/stream', {}, () => {}),
    ).rejects.toThrow('401')
  })
})
