package com.amelia.bridge

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import okio.Buffer
import java.nio.charset.StandardCharsets

internal object AmeliaNetHelpers {
    fun readBodyUtf8(body: RequestBody?): String {
        if (body == null) return ""
        return runCatching {
            val buf = Buffer()
            body.writeTo(buf)
            buf.readString(StandardCharsets.UTF_8)
        }.getOrDefault("")
    }

    fun jsonResponse(req: Request, body: String): Response {
        val mt = "application/json; charset=utf-8".toMediaType()
        val rb = body.toResponseBody(mt)
        return Response.Builder()
            .request(req)
            .protocol(Protocol.HTTP_1_1)
            .code(200)
            .message("OK")
            .body(rb)
            .build()
    }
}
