class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        (application as AIApplication).let { app ->
            if (!app.isSystemReady()) {
                showLoadingScreen()
                observeSystemStatus(app)
            }
        }
    }

    private fun observeSystemStatus(app: AIApplication) {
        val statusObserver = object : LifecycleObserver {
            @OnLifecycleEvent(Lifecycle.Event.ON_RESUME)
            fun checkStatus() {
                if (app.isSystemReady()) {
                    lifecycle.removeObserver(this)
                    proceedWithApp()
                }
            }
        }
        lifecycle.addObserver(statusObserver)
    }
}
