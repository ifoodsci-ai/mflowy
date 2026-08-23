-- /traces 验证：X-Telemetry-Token → Talos apiKeys:verify（credential 走 JSON body）

local cjson = require("cjson.safe")

local token = ngx.var.http_x_telemetry_token
if not token or #token == 0 then
    ngx.status = 401
    ngx.say(cjson.encode({error = "X-Telemetry-Token required"}))
    return
end

local res = ngx.location.capture("/_talos_admin/v2alpha1/admin/apiKeys:verify", {
    method = ngx.HTTP_POST,
    body = cjson.encode({credential = token}),
})

local data = res.body and cjson.decode(res.body) or nil
if res.status == 200 and data and data.is_valid then
    return  -- 验证通过，放行至 otelite
end

ngx.status = 401
ngx.say(cjson.encode({error = "invalid telemetry token",
                      code = data and data.error_code or nil}))
