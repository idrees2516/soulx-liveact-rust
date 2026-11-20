"""End-to-end test against the public LiveAct-RS deployment:
connects over WSS, starts a session, receives chunk events, viseme
timeline and binary audio frames; reports timing and stats."""
import asyncio, json, time, sys
import websockets

URL = "wss://8080-agq7tznykj3vpll2i2zm63cmjm.instabox.tech/ws"

async def main():
    async with websockets.connect(URL, max_size=64 * 1024 * 1024) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), 30))
        assert hello["type"] == "hello", hello
        print(f"hello: model={hello['model']} voices={len(hello['voices'])} rate={hello['sample_rate']}")

        await ws.send(json.dumps({
            "type": "start",
            "script": "Hello from the public internet! This is LiveAct running on InstaBox.",
            "voice": "af_sky",
            "speed": 1.0,
            "segments": [],
        }))
        t0 = time.time()
        chunks = 0
        audio_bytes = 0
        viseme_spans = 0
        audio_ms = 0
        started = None
        while True:
            raw = await asyncio.wait_for(ws.recv(), 60)
            if isinstance(raw, bytes):
                n = int.from_bytes(raw[4:8], "little")
                audio_bytes += len(raw) - 8
                assert n == (len(raw) - 8) // 2, "frame length mismatch"
                continue
            msg = json.loads(raw)
            t = msg["type"]
            if t == "chunk_start":
                started = time.time()
                audio_ms += msg["audioMs"]
                viseme_spans += len(msg["visemes"])
                print(f"chunk {msg['idx']}: {msg['audioMs']}ms audio, {len(msg['visemes'])} viseme spans")
            elif t == "chunk_end":
                chunks += 1
                dt = (time.time() - started) * 1000 if started else -1
                print(f"  -> received in {dt:.0f} ms")
            elif t == "progress":
                print(f"progress: {msg['generated']}/{msg['total']} {msg['state']}")
            elif t == "done":
                print(f"DONE: {chunks} chunks, {audio_ms/1000:.1f}s audio, "
                      f"{audio_bytes/1024:.0f} KB PCM, {viseme_spans} visemes, RTF={msg['rtf']}")
                break
            elif t == "error":
                print("ERROR:", msg); sys.exit(1)
        print(f"wall: {time.time()-t0:.1f}s")

asyncio.run(main())
