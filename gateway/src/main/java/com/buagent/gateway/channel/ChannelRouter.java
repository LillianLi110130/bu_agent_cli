package com.buagent.gateway.channel;

public interface ChannelRouter {
    boolean send(String sessionKey, String content);
}
