package com.cmb.tg.tgai.infrastructure.common.rest.wecom.client;

import com.cmb.tg.tgai.infrastructure.common.rest.wecom.ApiGatewayUtil;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.WecomSyncProperties;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.request.CustomerBaseInfoRequestDTO;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.request.EmployeeInfoRequestDTO;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.request.GroupBaseInfoRequestDTO;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response.CustomerBaseInfoResponseDTO;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response.EmployeeInfoResponseDTO;
import com.cmb.tg.tgai.infrastructure.common.rest.wecom.dto.response.GroupBaseInfoResponseDTO;
import lombok.RequiredArgsConstructor;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;

@Component
@RequiredArgsConstructor
public class WecomSyncClient {

    private static final String CUSTOMER_BBK_ID = "100003";
    private static final int MAX_RETRY_TIMES = 1;

    private final RestTemplate restTemplate;

    private final ApiGatewayUtil apiGatewayUtil;

    private final WecomSyncProperties wecomSyncProperties;

    public GroupBaseInfoResponseDTO queryGroupBaseInfo(final String roomId) {
        GroupBaseInfoRequestDTO requestDTO = GroupBaseInfoRequestDTO.builder().groupId(roomId).build();
        return post(wecomSyncProperties.getGroupBaseInfoUrl(), requestDTO, GroupBaseInfoResponseDTO.class);
    }

    public CustomerBaseInfoResponseDTO queryCustomerBaseInfo(final String custId) {
        CustomerBaseInfoRequestDTO requestDTO = CustomerBaseInfoRequestDTO.builder()
                .bbkId(CUSTOMER_BBK_ID)
                .custId(custId)
                .build();
        return post(wecomSyncProperties.getCustomerBaseInfoUrl(), requestDTO, CustomerBaseInfoResponseDTO.class);
    }

    public EmployeeInfoResponseDTO queryEmployeeInfo(final String employeeId) {
        EmployeeInfoRequestDTO requestDTO = EmployeeInfoRequestDTO.builder().sapId(employeeId).build();
        return post(wecomSyncProperties.getEmployeeInfoUrl(), requestDTO, EmployeeInfoResponseDTO.class);
    }

    private <T> T post(final String url, final Object request, final Class<T> responseType) {
        HttpHeaders headers = apiGatewayUtil.buildHeaders(
                wecomSyncProperties.getAppId(),
                wecomSyncProperties.getPrivateKey(),
                wecomSyncProperties.getSecret(),
                wecomSyncProperties.getVerify());
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Object> requestEntity = new HttpEntity<Object>(request, headers);
        int retryTimes = 0;
        while (true) {
            try {
                ResponseEntity<T> response = restTemplate.exchange(url, HttpMethod.POST, requestEntity, responseType);
                return response.getBody();
            } catch (ResourceAccessException ex) {
                if (retryTimes >= MAX_RETRY_TIMES) {
                    throw ex;
                }
                retryTimes++;
            }
        }
    }
}
