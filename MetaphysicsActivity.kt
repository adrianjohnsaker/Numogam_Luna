class MetaphysicsActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMetaphysicsBinding
    private val viewModel: MetaphysicsViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMetaphysicsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupObservers()
        setupUI()
    }

    private fun setupObservers() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.uiState.collect { state ->
                    when (state) {
                        is MetaphysicsViewModel.UiState.Processing -> showLoading()
                        is MetaphysicsViewModel.UiState.Stable -> showStableState(state)
                        is MetaphysicsViewModel.UiState.HighIntegration -> showHighIntegration(state)
                        is MetaphysicsViewModel.UiState.Unstable -> showUnstableWarning(state)
                        is MetaphysicsViewModel.UiState.Error -> showError(state.message)
                        MetaphysicsViewModel.UiState.Idle -> resetUI()
                    }
                }
            }
        }

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.dimensionalVectors.collect { vectors ->
                    updateDimensionalVisualization(vectors)
                }
            }
        }
    }

    private fun setupUI() {
        binding.btnProcess.setOnClickListener {
            val input = mapOf(
                "archetype" to binding.etSymbolicInput.text.toString(),
                "intensity" to binding.sliderIntensity.value
            )
            viewModel.processInput(input)
        }
    }

    private fun updateDimensionalVisualization(vectors: Map<String, List<Double>>) {
        // Update your visualization components here
        binding.vectorChart.updateData(vectors)
    }

    private fun showLoading() {
        binding.progressBar.visible()
        binding.btnProcess.disable()
    }

    private fun showStableState(state: MetaphysicsViewModel.UiState.Stable) {
        binding.progressBar.gone()
        binding.btnProcess.enable()
        binding.tvStatus.text = "System Stability: ${state.state.stabilityCoefficient}"
    }

    // ... other UI update methods
}
