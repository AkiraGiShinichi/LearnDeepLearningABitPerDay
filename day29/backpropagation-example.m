%https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/

x = [1 4 5]
Wxh = [[0.1 0.2];[0.3 0.4];[0.5 0.6]]
bh = [0.5 0.5]
hin = x * Wxh + bh
hout = 1 ./ (1 + exp(-hin))
%1 ./ (1 + exp(-hin))

Who = [[0.7 0.8];[0.9 0.1]]
bo = bh
oin = hout * Who + bo
oout = 1./ (1 + exp(-oin))

t = [0.1 0.05]
e = 1/2 * (oout - t) .^ 2

dedoout = oout - t
dooutdoin = oout .* (1 - oout)
dedwho = ((dedoout .* dooutdoin)' * hout)'
dedbo = dedoout .* dooutdoin
sum(dedbo)

dedhout = dedoout .* dooutdoin * Who
dhoutdhin = hout .* (1 - hout)
dedwxh = ((dedhout .* dhoutdhin)' * x)'

endv = 0
endv += 1