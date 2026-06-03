package com.cmb.tg.tgai.controller.message;

import com.cmb.tg.tgai.service.message.dto.CompleteRequest;
import com.cmb.tg.tgai.service.message.dto.ProgressRequest;
import com.cmb.tg.tgai.service.message.dto.SendTextRequest;
import com.cmb.tg.tgai.service.message.dto.SimpleOkResponse;
import com.cmb.tg.tgai.service.message.dto.WorkerRequest;
import com.cmb.tg.tgai.service.worker.WorkerGatewayService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api/worker")
public class WorkerController {

    private static final Logger logger = LoggerFactory.getLogger(WorkerController.class);

    private final WorkerGatewayService workerGatewayService;

    public WorkerController(WorkerGatewayService workerGatewayService) {
        this.workerGatewayService = workerGatewayService;
    }

    @PostMapping("/online")
    public SimpleOkResponse online(@RequestBody WorkerRequest request) {
        SimpleOkResponse response = workerGatewayService.online(request);
        return response;
    }

    @PostMapping("/offline")
    public SimpleOkResponse offline(@RequestBody WorkerRequest request) {
        SimpleOkResponse response = workerGatewayService.offline(request);
        return response;
    }

    @GetMapping(path = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@RequestParam("worker_id") String workerId) {
        return workerGatewayService.stream(workerId);
    }

    @PostMapping("/complete")
    public SimpleOkResponse complete(@RequestBody CompleteRequest request) {
        logger.info(
            "Received complete request. workerId={}, finalStatus={}, errorCode={}, finalContentLength={}",
            request.getWorkerNo(),
            request.getFinalStatus(),
            request.getErrorCode(),
            request.getFinalContent() == null ? 0 : request.getFinalContent().length()
        );
        SimpleOkResponse response = workerGatewayService.complete(request);
        logger.info("Completed complete request. workerId={}, ok={}", request.getWorkerNo(), response.getOk());
        return response;
    }

    @PostMapping("/progress")
    public SimpleOkResponse progress(@RequestBody ProgressRequest request) {
        logger.info(
            "Received progress request. workerId={}, contentLength={}",
            request.getWorkerNo(),
            request.getContent() == null ? 0 : request.getContent().length()
        );
        return workerGatewayService.progress(request);
    }

    @PostMapping("/send_text")
    public SimpleOkResponse sendText(@RequestBody SendTextRequest request) {
        return workerGatewayService.sendText(request);
    }

    @PostMapping(value = "/upload_attachment", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public SimpleOkResponse uploadAttachment(@RequestParam("file") MultipartFile file) throws IOException {
        return workerGatewayService.uploadAttachment(file);
    }
}
