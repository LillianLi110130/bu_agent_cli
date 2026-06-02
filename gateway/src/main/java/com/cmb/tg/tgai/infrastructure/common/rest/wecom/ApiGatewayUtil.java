package com.cmb.tg.tgai.infrastructure.common.rest.wecom;

import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;

@Component
public class ApiGatewayUtil {

    public HttpHeaders buildHeaders(final String appId, final String privateKey, final String secret, final String verify) {
        String timestamp = String.valueOf(System.currentTimeMillis() / 1000);
        String sign = "";

        HttpHeaders headers = new HttpHeaders();
        headers.add("appid", appId);
        headers.add("timestamp", timestamp);
        headers.add("sign", sign);
        headers.add("apisign", buildApiSign(appId, secret, timestamp, sign));
        headers.add("verify", verify);
        return headers;
    }

    private String buildApiSign(final String appId, final String secret, final String timestamp, final String sign) {
        return "appid=" + appId + "&secret=" + secret + "&sign=" + sign + "&timestamp=" + timestamp;
    }
}
