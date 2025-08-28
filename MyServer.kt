import fi.iki.elonen.NanoHTTPD
import java.io.IOException

class MyServer(port: Int) : NanoHTTPD(port) {

    @Throws(IOException::class)
    fun startServer() {
        start(SOCKET_READ_TIMEOUT, false)
    }

    override fun serve(session: IHTTPSession): Response {
        val msg = "<html><body><h1>Hello from NanoHTTPD Server</h1></body></html>"
        return newFixedLengthResponse(msg)
    }
}
