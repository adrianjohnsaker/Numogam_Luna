
import com.antonio.my.ai.girlfriend.free.consciousness.amelia.phase4.Phase4ConsciousnessBridge
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object Phase4Module {
    
    @Provides
    @Singleton
    fun providePhase4ConsciousnessBridge(): Phase4ConsciousnessBridge {
        return Phase4ConsciousnessBridge()
    }
}
