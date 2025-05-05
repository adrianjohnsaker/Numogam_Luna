// DesireProductionBridge.kt
package com.antonio.my.ai.girlfriend.free

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class DesireProductionBridge private constructor(context: Context) {
    private val pythonBridge = PythonBridge.getInstance(context)
    
    companion object {
        @Volatile private var instance: DesireProductionBridge? = null
        
        fun getInstance(context: Context): DesireProductionBridge {
            return instance ?: synchronized(this) {
                instance ?: DesireProductionBridge(context).also { instance = it }
            }
        }
    }
    
    /**
     * Generate productive desire from context
     */
    suspend fun produceCreativeDesire(context: Map<String, Any>): DesireProductionResult? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "desire_production_engine",
                "produce_creative_desire",
                context
            )
            
            @Suppress("UNCHECKED_CAST")
            (result as? Map<String, Any>)?.let { map ->
                DesireProductionResult(
                    sourceFlow = map["source_flow"] as? String ?: "",
                    type = map["type"] as? String ?: "",
                    intensity = map["intensity"] as? Double ?: 0.0,
                    creativeVector = map["creative_vector"] as? Map<String, Double>,
                    actualizations = map["actualizations"] as? List<Map<String, Any>>,
                    deterritorialization = map["deterritorialization"] as? Map<String, Any>,
                    reterritorialization = map["reterritorialization"] as? List<Map<String, Any>>
                )
            }
        }
    }
    
    /**
     * Assemble desiring machine from context
     */
    suspend fun assembleDesiring�# scauwjh/Memos
# Memos/src/main/java/com/wjh/memos/web/controllers/UploadController.java
package com.wjh.memos.web.controllers;

import com.wjh.memos.config.AppConfig;
import com.wjh.memos.dto.ResultDTO;
import com.wjh.memos.dto.UploadDTO;
import com.wjh.memos.entity.Image;
import com.wjh.memos.entity.ImageGroup;
import com.wjh.memos.exception.CommonException;
import com.wjh.memos.exception.ExceptionCode;
import com.wjh.memos.service.ImageGroupService;
import com.wjh.memos.service.ImageService;
import com.wjh.memos.service.UserService;
import com.wjh.memos.utils.CsvUtil;
import com.wjh.memos.utils.EncryptUtil;
import com.wjh.memos.utils.FileUtil;
import com.wjh.memos.utils.ImageUtil;
import com.wjh.memos.utils.UUIDGenerator;
import com.wjh.memos.web.AbstractController;
import freemarker.ext.beans.StringModel;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

/**
 * Created by wjh on 16-4-17.
 */
@Controller
@RequestMapping(value = "upload")
public class UploadController extends AbstractController {
    @Autowired
    private ImageService imageService;
    @Autowired
    private UserService userService;
    @Autowired
    private ImageGroupService imageGroupService;
    @Autowired
    private AppConfig appConfig;

    @RequestMapping(value = "image", method = RequestMethod.GET)
    public String uploadIndex(Model model) {
        String uuid = getCurrentUser().getUid();
        List<ImageGroup> imageGroupList = imageGroupService.findByUserUid(uuid);
        model.addAttribute("imageGroupList", imageGroupList);
        return "upload/image";
    }

    @RequestMapping(value = "image", method = RequestMethod.POST)
    @ResponseBody
    public ResultDTO uploadImage(@RequestParam("file") MultipartFile multipartFile,
                               String groupId,
                               RedirectAttributes redirectAttributes) {
        ResultDTO result = new ResultDTO();
        try {
            if (multipartFile.isEmpty()) {
                throw new CommonException(ExceptionCode.NO_FILE);
            }
            String fileName = multipartFile.getOriginalFilename();
            String fileType = fileName.substring(fileName.lastIndexOf("."));
            String extension = FilenameUtils.getExtension(fileName);
            String uuid = getCurrentUser().getUid();
            String path = FileUtil.createFilePath(appConfig.getUploadPath() + fileName, String.valueOf(System.currentTimeMillis()));

            // create directory
            File targetDir = new File(appConfig.getUploadPath());
            if (!targetDir.exists()) {
                targetDir.mkdir();
            }

            // copy file
            File targetFile = new File(path);
            multipartFile.transferTo(targetFile);

            String md5 = EncryptUtil.getFileMD5(targetFile);
            Image image = imageService.findByMd5(md5);
            if (image == null) {
                // Create object model file and save it
                image = new Image();
                image.setUserUid(uuid);
                image.setOriginName(fileName);
                image.setMd5(md5);

                try {
                    // get image's width and height
                    BufferedImage in = ImageIO.read(targetFile);
                    image.setImgWidth(in.getWidth());
                    image.setImgHeight(in.getHeight());
                } catch (Exception e) {
                    logger.warn("Unexpected file format", e);
                }
                image.setRealPath(path);

                // create a thumbnail
                try {
                    String thumbnailPath = FileUtil.createFilePath(appConfig.getUploadPath() + "thumbnails/"
                            + fileName, "");
                    File thumbnailDir = new File(appConfig.getUploadPath() + "thumbnails/");
                    if (!thumbnailDir.exists()) {
                        thumbnailDir.mkdir();
                    }

                    // create a thumbnail
                    BufferedImage thumbnail = ImageUtil.resize(targetFile, Image.THUMBNAIL_WIDTH, Image.THUMBNAIL_WIDTH, true);
                    FileUtil.saveImage(thumbnail, thumbnailPath, extension);
                    image.setThumbnailPath(thumbnailPath);
                } catch (Exception e) {
                    logger.error("Can not create a thumbnail for file = " + fileName, e);
                }

                // save image metadata
                image.setId(UUIDGenerator.getUUID());
                image.setCreateTime(new Date());
                image.setStatus(Image.STATUS_NORMAL);
                String dateDir = new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                image.setShowPath(dateDir + "/" + fileName);
                imageService.save(image);
            }

            // save relationship between group and image
            if (StringUtils.isNotBlank(groupId)) {
                imageGroupService.addImage(uuid, groupId, image.getId());
                //add id, so that javascript can append to image list
                result.put("groupId", groupId);
            }
            result.put("image", image);
            result.setCode(ExceptionCode.SUCCESS_CODE);
        } catch (Exception e) {
            logger.error("Error during uploading image", e);
            result.setCode(e instanceof CommonException
                    ? ((CommonException) e).getErrorCode()
                    : ExceptionCode.UNKNOWN_ERROR);
            result.setMsg(e.getMessage());
        }
        return result;
    }

    @RequestMapping(value = "csv/{type}", method = RequestMethod.POST)
    @ResponseBody
    public ResultDTO uploadCsv(@RequestParam("file") MultipartFile multipartFile,
                             @PathVariable String type,
                             RedirectAttributes redirectAttributes) {
        ResultDTO result = new ResultDTO();
        try {
            if (multipartFile.isEmpty()) {
                throw new CommonException(ExceptionCode.NO_FILE);
            }
            String fileName = multipartFile.getOriginalFilename();
            String extension = FilenameUtils.getExtension(fileName);
            if (!extension.equals("csv")) {
                throw new CommonException(ExceptionCode.BAD_FILE_TYPE);
            }

            // create directory
            File targetDir = new File(appConfig.getUploadPath());
            if (!targetDir.exists()) {
                targetDir.mkdir();
            }

            // copy file
            String path = appConfig.getUploadPath() + fileName;
            File targetFile = new File(path);
            multipartFile.transferTo(targetFile);

            // check file type
            // content type
            UploadDTO processReasult = null;
            if ("account".equals(type)) {
                processReasult = CsvUtil.processAccount(targetFile);
            } else if ("category".equals(type)) {
                processReasult = CsvUtil.processCategory(targetFile);
            } else {
                throw new CommonException(ExceptionCode.BAD_FILE_TYPE);
            }
            result.put("processReasult", processReasult);
            result.setCode(ExceptionCode.SUCCESS_CODE);
        } catch (Exception e) {
            logger.error("Error during uploading image", e);
            result.setCode(e instanceof CommonException
                    ? ((CommonException) e).getErrorCode()
                    : ExceptionCode.UNKNOWN_ERROR);
            result.setMsg(e.getMessage());
        }
        return result;
    }
}
�# hj-core/kafka-streams-examples
/*
 * Copyright Confluent Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.confluent.examples.streams.streamdsl.stateless.stream;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.kstream.KStream;

import java.util.Properties;

/**
 * Demonstrates how to merge two KStreams (with same key type and same value type).
 * Same as calling StreamsBuilder.stream(...).merge(otherStream) which is illustrated
 * in the `MergeStreamsWithKeyValueTypesExample`.
 *
 * Use case: You have two sources of input data that you want to process together in
 * the same Kafka Streams pipeline.
 *
 * In this example we use two topics and we merge both.
 *
 * Note: This example uses lambda expressions and thus works with Java 8+ only.
 */
public class MergeKStreamsExample {

  public static void main(final String[] args) {
    final String bootstrapServers = args.length > 0 ? args[0] : "localhost:9092";
    final Properties streamsConfiguration = new Properties();
    // Give the Streams application a unique name.  The name must be unique in the Kafka cluster
    // against which the application is run.
    streamsConfiguration.put(StreamsConfig.APPLICATION_ID_CONFIG, "merge-kstreams-example");
    streamsConfiguration.put(StreamsConfig.CLIENT_ID_CONFIG, "merge-kstreams-example-client");
    // Where to find Kafka broker(s).
    streamsConfiguration.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
    // Specify default (de)serializers for record keys and for record values.
    streamsConfiguration.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass().getName());
    streamsConfiguration.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass().getName());
    // Records should be flushed every 10 seconds. This is less than the default
    // in order to keep this example interactive.
    streamsConfiguration.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 10 * 1000);

    final StreamsBuilder builder = new StreamsBuilder();

    // First stream
    final KStream<String, String> stream1 = builder.stream("stream1");

    // Variant 1: Use StreamsBuilder.stream(), which will
        /**
     * Assemble desiring machine from context
     */
    suspend fun assembleDesiringMachine(context: Map<String, Any>): DesiringMachineResult? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "desire_production_engine",
                "assemble_desiring_machine",
                context
            )
            
            @Suppress("UNCHECKED_CAST")
            (result as? Map<String, Any>)?.let { map ->
                DesiringMachineResult(
                    machineType = map["machine_type"] as? String ?: "",
                    components = map["components"] as? List<String> ?: emptyList(),
                    connections = map["connections"] as? Map<String, String> ?: emptyMap(),
                    flowIntensity = map["flow_intensity"] as? Double ?: 0.0,
                    deterritorializationFactor = map["deterritorialization_factor"] as? Double ?: 0.0,
                    reterritorializationVectors = map["reterritorialization_vectors"] as? List<Map<String, Any>> ?: emptyList()
                )
            }
        }
    }

    /**
     * Analyze desire flows between machines
     */
    suspend fun analyzeDesireFlows(sourceMachine: String, targetMachine: String): FlowAnalysisResult? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "desire_production_engine",
                "analyze_desire_flows",
                mapOf(
                    "source_machine" to sourceMachine,
                    "target_machine" to targetMachine
                )
            )
            
            @Suppress("UNCHECKED_CAST")
            (result as? Map<String, Any>)?.let { map ->
                FlowAnalysisResult(
                    flowType = map["flow_type"] as? String ?: "",
                    intensity = map["intensity"] as? Double ?: 0.0,
                    resistance = map["resistance"] as? Double ?: 0.0,
                    potential = map["potential"] as? Double ?: 0.0,
                    flowComponents = map["flow_components"] as? List<String> ?: emptyList(),
                    blockages = map["blockages"] as? List<String> ?: emptyList()
                )
            }
        }
    }

    /**
     * Calculate desire production capacity
     */
    suspend fun calculateProductionCapacity(machineId: String): ProductionCapacityResult? {
        return withContext(Dispatchers.IO) {
            val result = pythonBridge.executeFunction(
                "desire_production_engine",
                "calculate_production_capacity",
                mapOf("machine_id" to machineId)
            )
            
            @Suppress("UNCHECKED_CAST")
            (result as? Map<String, Any>)?.let { map ->
                ProductionCapacityResult(
                    machineId = machineId,
                    maxCapacity = map["max_capacity"] as? Double ?: 0.0,
                    currentUtilization = map["current_utilization"] as? Double ?: 0.0,
                    efficiency = map["efficiency"] as? Double ?: 0.0,
                    bottlenecks = map["bottlenecks"] as? List<String> ?: emptyList()
                )
            }
        }
    }

    /**
     * Optimize desire production flow
     */
    suspend fun optimizeFlow(machineId: String, parameters: Map<String, Any>): FlowOptimizationResult? {
        return withContext(Dispatchers.IO) {
            val context = mutableMapOf("machine_id" to machineId)
            context.putAll(parameters)
            
            val result = pythonBridge.executeFunction(
                "desire_production_engine",
                "optimize_flow",
                context
            )
            
            @Suppress("UNCHECKED_CAST")
            (result as? Map<String, Any>)?.let { map ->
                FlowOptimizationResult(
                    machineId = machineId,
                    originalEfficiency = map["original_efficiency"] as? Double ?: 0.0,
                    optimizedEfficiency = map["optimized_efficiency"] as? Double ?: 0.0,
                    recommendedAdjustments = map["recommended_adjustments"] as? List<String> ?: emptyList(),
                    potentialGain = map["potential_gain"] as? Double ?: 0.0
                )
            }
        }
    }
}

// Data classes for the results
data class DesireProductionResult(
    val sourceFlow: String,
    val type: String,
    val intensity: Double,
    val creativeVector: Map<String, Double>?,
    val actualizations: List<Map<String, Any>>?,
    val deterritorialization: Map<String, Any>?,
    val reterritorialization: List<Map<String, Any>>?
)

data class DesiringMachineResult(
    val machineType: String,
    val components: List<String>,
    val connections: Map<String, String>,
    val flowIntensity: Double,
    val deterritorializationFactor: Double,
    val reterritorializationVectors: List<Map<String, Any>>
)

data class FlowAnalysisResult(
    val flowType: String,
    val intensity: Double,
    val resistance: Double,
    val potential: Double,
    val flowComponents: List<String>,
    val blockages: List<String>
)

data class ProductionCapacityResult(
    val machineId: String,
    val maxCapacity: Double,
    val currentUtilization: Double,
    val efficiency: Double,
    val bottlenecks: List<String>
)

data class FlowOptimizationResult(
    val machineId: String,
    val originalEfficiency: Double,
    val optimizedEfficiency: Double,
    val recommendedAdjustments: List<String>,
    val potentialGain: Double
)
