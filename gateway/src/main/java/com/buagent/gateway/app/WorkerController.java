package com.buagent.gateway.app;

import com.buagent.gateway.app.dto.CompleteRequest;
import com.buagent.gateway.app.dto.PollRequest;
import com.buagent.gateway.app.dto.PollResponse;
import com.buagent.gateway.app.dto.RenewRequest;
import com.buagent.gateway.app.dto.SendTextRequest;
import com.buagent.gateway.app.dto.SimpleOkResponse;
import org.springframework.http.MediaType;
import com.buagent.gateway.worker.WorkerGatewayService;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api/worker")
public class WorkerController {

    private final WorkerGatewayService workerGatewayService;

    public WorkerController(WorkerGatewayService workerGatewayService) {
        this.workerGatewayService = workerGatewayService;
    }

    @PostMapping("/poll")
    public DeferredResult<PollResponse> poll(@RequestBody PollRequest request) {
        return workerGatewayService.poll(request);
    }

    @PostMapping("/online")
    public SimpleOkResponse online(@RequestBody PollRequest request) {
        return workerGatewayService.online(request.getWorkerId());
    }

    @PostMapping("/offline")
    public SimpleOkResponse offline(@RequestBody PollRequest request) {
        return workerGatewayService.offline(request.getWorkerId());
    }

    @PostMapping("/renew")
    public SimpleOkResponse renew(@RequestBody RenewRequest request) {
        return workerGatewayService.renew(request);
    }

    @PostMapping("/complete")
    public SimpleOkResponse complete(@RequestBody CompleteRequest request) {
        return workerGatewayService.complete(request);
    }

    @PostMapping("/send_text")
    public SimpleOkResponse sendText(@RequestBody SendTextRequest request) {
        return workerGatewayService.sendText(request);
    }

    @PostMapping(value = "/upload_attachment", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public SimpleOkResponse uploadAttachment(
        @RequestParam(value = "session_key", required = false) String sessionKey,
        @RequestParam("worker_id") String workerId,
        @RequestParam(value = "mime_type", required = false) String mimeType,
        @RequestParam(value = "file_size", required = false) Long fileSize,
        @RequestParam("file") MultipartFile file
    ) throws IOException {
        return workerGatewayService.uploadAttachment(sessionKey, workerId, mimeType, fileSize, file);
    }
}
