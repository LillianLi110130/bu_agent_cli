package com.buagent.gateway.channel;

public interface ChannelRouter {
    boolean send(String sessionKey, String content);

    boolean sendAttachment(
        String sessionKey,
        String fileName,
        String mimeType,
        Long fileSize,
        String fileContentBase64
    );
}
