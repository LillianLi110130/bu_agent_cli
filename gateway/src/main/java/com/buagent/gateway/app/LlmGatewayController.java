package com.buagent.gateway.app;

import com.buagent.gateway.app.dto.LlmQueryRequest;
import com.buagent.gateway.llm.LlmGatewayService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.CacheControl;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;
import reactor.core.publisher.Flux;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

@RestController
public class LlmGatewayController {

    private static final Logger logger = LoggerFactory.getLogger(LlmGatewayController.class);

    private final LlmGatewayService llmGatewayService;

    public LlmGatewayController(LlmGatewayService llmGatewayService) {
        this.llmGatewayService = llmGatewayService;
    }

    @PostMapping(path = "/llm/query-stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public ResponseEntity<StreamingResponseBody> queryStream(
        @RequestBody LlmQueryRequest request,
        @RequestHeader HttpHeaders headers
    ) {
        StreamingResponseBody responseBody = outputStream -> {
            ByteArrayOutputStream rawResponseBuffer = new ByteArrayOutputStream();
            Flux<DataBuffer> upstream = llmGatewayService.queryStream(request, headers);
            try {
                upstream.doOnNext(dataBuffer -> {
                    try {
                        byte[] bytes = new byte[dataBuffer.readableByteCount()];
                        dataBuffer.read(bytes);
                        outputStream.write(bytes);
                        outputStream.flush();
                        rawResponseBuffer.write(bytes, 0, bytes.length);
                    } catch (IOException exception) {
                        throw new IllegalStateException(
                            "Failed to write LLM stream response",
                            exception
                        );
                    } finally {
                        DataBufferUtils.release(dataBuffer);
                    }
                }).blockLast();
            } finally {
                logger.info(
                    "LLM stream raw response: {}",
                    rawResponseBuffer.toString(StandardCharsets.UTF_8.name())
                );
            }
        };

        return ResponseEntity.ok()
            .contentType(MediaType.TEXT_EVENT_STREAM)
            .cacheControl(CacheControl.noCache())
            .header("X-Accel-Buffering", "no")
            .body(responseBody);
    }
}
