function inputChar(char) {
  document.getElementById("field1").value += char;
}

function delChar() {
    let val = document.getElementById("field1").value;
    document.getElementById("field1").value = val.slice(0,-1);
}

