#!/usr/bin/env python3
"""
Nanobot Co-Browse Relay Server
Bridges Chrome extension (CDP) with nanobot agent tools.
Port: 18793 loopback only.
"""
import asyncio
import json
import os
import time
from aiohttp import web, WSMsgType

RELAY_PORT = int(os.environ.get("NANOBOT_RELAY_PORT", "18793"))
RELAY_TOKEN = os.environ.get("NANOBOT_RELAY_TOKEN", "")

state = {
    "connected": False,
    "tabs": {},
    "last_url": "",
    "last_title": "",
    "event_buffer": [],
    "ws": None,
    "pending": {},
}


def _check_token(request):
    if not RELAY_TOKEN:
        return True
    token = (
        request.rel_url.query.get("token", "")
        or request.headers.get("x-nanobot-relay-token", "")
    )
    return token == RELAY_TOKEN


async def handle_root(request):
    return web.Response(text="Nanobot Browser Relay OK")


async def handle_version(request):
    if not _check_token(request):
        return web.Response(status=401, text="Unauthorized")
    return web.json_response({
        "Browser": "Nanobot Relay", "version": "1.0.0",
        "connected": state["connected"], "tabs": len(state["tabs"]),
        "last_url": state["last_url"],
    })


async def handle_state(request):
    if not _check_token(request):
        return web.Response(status=401, text="Unauthorized")
    return web.json_response({
        "connected": state["connected"],
        "tabs": state["tabs"],
        "last_url": state["last_url"],
        "last_title": state["last_title"],
        "tab_count": len(state["tabs"]),
        "recent_events": state["event_buffer"][-10:],
    })


async def handle_command(request):
    if not _check_token(request):
        return web.Response(status=401, text="Unauthorized")
    if not state["connected"] or not state["ws"]:
        return web.json_response({"error": "No extension connected"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON")
    cmd_id = int(time.time() * 1000) % 2147483647
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    state["pending"][cmd_id] = fut
    try:
        await state["ws"].send_str(json.dumps({
            "id": cmd_id,
            "method": "forwardCDPCommand",
            "params": {
                "method": body.get("method", ""),
                "params": body.get("params", {}),
                "sessionId": body.get("sessionId"),
            },
        }))
        result = await asyncio.wait_for(fut, timeout=10.0)
        return web.json_response({"result": result})
    except asyncio.TimeoutError:
        state["pending"].pop(cmd_id, None)
        return web.json_response({"error": "Command timed out"}, status=504)
    except Exception as e:
        state["pending"].pop(cmd_id, None)
        return web.json_response({"error": str(e)}, status=500)


async def handle_websocket(request):
    if not _check_token(request):
        return web.Response(status=401, text="Unauthorized")
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    state["connected"] = True
    state["ws"] = ws
    print("[relay] Extension connected")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                method = data.get("method")
                if method == "ping":
                    await ws.send_str(json.dumps({"method": "pong"}))
                    continue
                if method == "forwardCDPEvent":
                    params = data.get("params", {})
                    evt = params.get("method", "")
                    ep = params.get("params", {})
                    if evt == "Target.attachedToTarget":
                        target = ep.get("targetInfo", {})
                        sid = ep.get("sessionId", "")
                        state["tabs"][sid] = {
                            "sessionId": sid,
                            "targetId": target.get("targetId", ""),
                            "url": target.get("url", ""),
                            "title": target.get("title", ""),
                        }
                        state["last_url"] = target.get("url", state["last_url"])
                        state["last_title"] = target.get("title", state["last_title"])
                    elif evt == "Target.detachedFromTarget":
                        state["tabs"].pop(ep.get("sessionId", ""), None)
                    elif evt == "Page.frameNavigated":
                        frame = ep.get("frame", {})
                        if not frame.get("parentId"):
                            url = frame.get("url", "")
                            if url and not url.startswith("about:"):
                                state["last_url"] = url
                                for tab in state["tabs"].values():
                                    tab["url"] = url
                    state["event_buffer"].append({"method": evt, "ts": time.time()})
                    if len(state["event_buffer"]) > 50:
                        state["event_buffer"] = state["event_buffer"][-50:]
                    continue
                cmd_id = data.get("id")
                if cmd_id is not None and cmd_id in state["pending"]:
                    fut = state["pending"].pop(cmd_id)
                    if not fut.done():
                        if "error" in data:
                            fut.set_exception(Exception(str(data["error"])))
                        else:
                            fut.set_result(data.get("result"))
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
    finally:
        state["connected"] = False
        state["ws"] = None
        state["tabs"].clear()
        for fut in state["pending"].values():
            if not fut.done():
                fut.cancel()
        state["pending"].clear()
        print("[relay] Extension disconnected")
    return ws


def make_app():
    app = web.Application()
    app.router.add_route("HEAD", "/", handle_root)
    app.router.add_route("GET", "/", handle_root)
    app.router.add_get("/json/version", handle_version)
    app.router.add_get("/state", handle_state)
    app.router.add_post("/command", handle_command)
    app.router.add_get("/extension", handle_websocket)
    return app


if __name__ == "__main__":
    print(f"[relay] Nanobot Browser Relay starting on 127.0.0.1:{RELAY_PORT}")
    web.run_app(make_app(), host="127.0.0.1", port=RELAY_PORT, print=None)