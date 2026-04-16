package com.buagent.gateway.worker;

import com.buagent.gateway.app.dto.CompleteRequest;
import com.buagent.gateway.app.dto.DebugInboundRequest;
import com.buagent.gateway.app.dto.PollMessageDto;
import com.buagent.gateway.app.dto.PollRequest;
import com.buagent.gateway.app.dto.PollResponse;
import com.buagent.gateway.app.dto.RenewRequest;
import com.buagent.gateway.app.dto.SimpleOkResponse;
import com.buagent.gateway.channel.ChannelRouter;
import com.buagent.gateway.session.InFlightDelivery;
import com.buagent.gateway.session.InboundMessageSnapshot;
import com.buagent.gateway.session.PollWaiter;
import com.buagent.gateway.session.SessionMailbox;
import com.buagent.gateway.session.SessionRegistry;
import com.buagent.gateway.store.entity.InboundMessageEntity;
import com.buagent.gateway.store.entity.OutboundMessageEntity;
import com.buagent.gateway.store.entity.SessionStateEntity;
import com.buagent.gateway.store.mapper.InboundMessageMapper;
import com.buagent.gateway.store.mapper.OutboundMessageMapper;
import com.buagent.gateway.store.mapper.SessionStateMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.web.server.ResponseStatusException;

import java.util.Collections;
import java.util.UUID;

@Service
public class WorkerGatewayService {

    private static final Logger logger = LoggerFactory.getLogger(WorkerGatewayService.class);

    private final SessionRegistry sessionRegistry;
    private final SessionStateMapper sessionStateMapper;
    private final InboundMessageMapper inboundMessageMapper;
    private final OutboundMessageMapper outboundMessageMapper;
    private final ChannelRouter channelRouter;
    private final long deliveryLeaseTimeoutMillis;
    private final long ownerActiveTimeoutMillis;
    private final long pollWaitTimeoutMillis;

    public WorkerGatewayService(
        SessionRegistry sessionRegistry,
        SessionStateMapper sessionStateMapper,
        InboundMessageMapper inboundMessageMapper,
        OutboundMessageMapper outboundMessageMapper,
        ChannelRouter channelRouter,
        @Value("${gateway.delivery-lease-timeout-seconds:120}") long deliveryLeaseTimeoutSeconds,
        @Value("${gateway.owner-active-timeout-seconds:90}") long ownerActiveTimeoutSeconds,
        @Value("${gateway.poll-wait-timeout-ms:25000}") long pollWaitTimeoutMillis
    ) {
        this.sessionRegistry = sessionRegistry;
        this.sessionStateMapper = sessionStateMapper;
        this.inboundMessageMapper = inboundMessageMapper;
        this.outboundMessageMapper = outboundMessageMapper;
        this.channelRouter = channelRouter;
        this.deliveryLeaseTimeoutMillis = deliveryLeaseTimeoutSeconds * 1000L;
        this.ownerActiveTimeoutMillis = ownerActiveTimeoutSeconds * 1000L;
        this.pollWaitTimeoutMillis = pollWaitTimeoutMillis;
    }

    public SimpleOkResponse acceptDebugInbound(DebugInboundRequest request) {
        InboundMessageEntity inboundMessageEntity = new InboundMessageEntity();
        inboundMessageEntity.setSessionKey(request.getSessionKey());
        inboundMessageEntity.setEpoch(epoch);
        inboundMessageEntity.setContent(request.getContent());
        inboundMessageEntity.setStatus("RECEIVED");
        inboundMessageEntity.setCreatedAt(System.currentTimeMillis());
        inboundMessageMapper.insert(inboundMessageEntity);


        return new SimpleOkResponse(true);
    }

    public SimpleOkResponse online(PollRequest request) {
        // 将online_worker表中worker_id=request.workerId的记录更新为online
    }

    public SimpleOkResponse offline(PollRequest request) {
        // 将online_worker表中worker_id=request.workerId的记录更新为offline
    }


    public SimpleOkResponse poll(PollRequest request) {
        // 从数据库查最老的一条RECEIVed状态且sessionkey由本次request构造的入站消息
        // 更新状态为CONSUMED并返回
    }


    public SimpleOkResponse complete(CompleteRequest request) {
        OutboundMessageEntity outboundMessageEntity = new OutboundMessageEntity();
        outboundMessageEntity.setSessionKey(request.getSessionKey());
        outboundMessageEntity.setEpoch(inFlightDelivery.getEpoch());
        outboundMessageEntity.setInboundMessageId(inFlightDelivery.getMessageId());
        outboundMessageEntity.setContent(request.getFinalContent());
        outboundMessageEntity.setStatus("PENDING");
        outboundMessageEntity.setCreatedAt(System.currentTimeMillis());
        outboundMessageMapper.insert(outboundMessageEntity);

        boolean sent = channelRouter.send(request.getSessionKey(), request.getFinalContent());
        outboundMessageMapper.updateStatus(outboundMessageEntity.getId(), sent ? "SENT" : "FAILED");
        // 发送给远程Channel
        return new SimpleOkResponse(true);
        }
    }
}
