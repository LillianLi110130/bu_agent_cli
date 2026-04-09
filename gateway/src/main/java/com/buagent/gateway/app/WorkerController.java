package com.buagent.gateway.app;

import com.buagent.gateway.app.dto.CompleteRequest;
import com.buagent.gateway.app.dto.PollRequest;
import com.buagent.gateway.app.dto.PollResponse;
import com.buagent.gateway.app.dto.RenewRequest;
import com.buagent.gateway.app.dto.SendTextRequest;
import com.buagent.gateway.app.dto.SimpleOkResponse;
import com.buagent.gateway.app.dto.UploadAttachmentRequest;
import com.buagent.gateway.worker.WorkerGatewayService;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.context.request.async.DeferredResult;

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

    @PostMapping("/upload_attachment")
    public SimpleOkResponse uploadAttachment(@RequestBody UploadAttachmentRequest request) {
        return workerGatewayService.uploadAttachment(request);
    }
}
