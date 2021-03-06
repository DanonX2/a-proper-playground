const grid = [10,10];
const canvasx = grid[0]*50;
const canvasy = grid[1]*50;
var foods = [];
var bots = [];
var generation = [];
var gencounter = 0;
var worldmap = new Array(grid[0]*grid[1]).fill(0);
const arrSum = arr => arr.reduce((a,b) => a + b, 0)
var fps = 5;
var mutationrate = 0;
var nnsize = [10,5];
var learningrate = 0.05;


function setup() {
  createCanvas(canvasx,canvasy);

  button1 = createButton('Add Food');
  button1.position(8, canvasy+16)
  button1.class('button');
  button1.mouseClicked(addfood);

  button2 = createButton('Add Random Bot');
  button2.position(150, canvasy+16)
  button2.class('button');
  button2.mouseClicked(addbot);

  button3 = createButton('Add 10 Random Bot');
  button3.position(358, canvasy+16)
  button3.class('button');
  button3.mouseClicked(add10bot);
  
  button5 = createButton('Add Inherit Bot');
  button5.position(150, canvasy+63);
  button5.class('button');
  button5.mouseClicked(addIbot);

  button6 = createButton('Add 10 Inherit Bot');
  button6.position(358, canvasy+63);
  button6.class('button');
  button6.mouseClicked(add10Ibot);

  button4 = createButton('Next Gen');
  button4.position(515, 8);
  button4.class('button');
  button4.mouseClicked(nextGen);

  input1 = createInput();
  input1.position(515, 90);
  input1.mouseClicked(updatefps);

  button7 = createButton('Next Auto Gen');
  button7.position(656, 8);
  button7.class('button');
  button7.mouseClicked(nextAutoGen);

  infoboard = createElement('p','testing').position(516,90);

  noStroke();
}
//////////////////////////////////////////////////////////////////////////
function draw() {

  frameRate(fps);
  background('#b3b3b3');
  worldrun1();

  for (i in foods) {
    foods[i].show();
  }

  for (i0 in bots) {
    bots[i0].move();
    bots[i0].show();
    bots[i0].brain.learn(learningrate);
  }

  infoupdate(); //updates info board
  worldrun2(); //updates worldmap

}
//////////////////////////////////////////////////////////////////////////
function food() {
  this.init = function() {
    this.x = Math.floor(Math.random() * grid[0]);
    this.y = Math.floor(Math.random() * grid[1]);
    this.location = [this.x*50,this.y*50];
  }
  this.show = function() {
    fill('red')
    rect(this.location[0],this.location[1],50,50);
  }
}

function bot() {
  this.init = function() {
    this.x = Math.floor(Math.random() * grid[0]);
    this.y = Math.floor(Math.random() * grid[1]);
    this.location = [this.x*50,this.y*50];
    this.hunger = 100;
    this.botinput = worldmap.slice();
    this.botinput.push(this.x);
    this.botinput.push(this.y);
    this.brain = new network()
    this.brain.init(nnsize)
  }
  this.show = function() {
    fill('black')
    rect(this.location[0],this.location[1],50,50);
  }
  this.update = function() {
    this.botinput = worldmap.slice();
    this.botinput.push(this.x);
    this.botinput.push(this.y);
    this.brain.loaddata(this.botinput);
    this.brain.fp();
  }
  this.action = function() {
    this.update();
    this.actionid = indexOfMax(this.brain.output);
    if (this.actionid == 0 && this.x < 9) {
      this.x += 1;
      this.hunger -= 5;
      this.brain.reward = -1;
      this.brain.totalreward -= 1;
    }
    else if (this.actionid == 1 && this.x > 0) {
      this.x -= 1;
      this.hunger -= 5;
      this.brain.reward = -1;
      this.brain.totalreward -= 1;
    }
    else if (this.actionid == 2 && this.y < 9) {
      this.y += 1;
      this.hunger -= 5;
      this.brain.reward = -1;
      this.brain.totalreward -= 1;
    }
    else if (this.actionid == 3 && this.y > 0) {
      this.y -= 1;
      this.hunger -= 5;
      this.brain.reward = -1;
      this.brain.totalreward -= 1;
    }
    else if (this.actionid == 4) {
      this.brain.reward = -1
      this.brain.totalreward -= 1;
    
    }
  }
  this.move = function() {
    this.action();
    this.location = [this.x*50,this.y*50];
    this.hunger -= 5;
  }
}

function addfood() {
  for (i00=0;i00<10;i00++) {
    foods.push(new food());
    foods[foods.length-1].init();
  }
  delete i00;
}

function addbot() {
  bots.push(new bot());
  bots[bots.length-1].init();
}

function add10bot() {
  for (i00=0;i00<10;i00++) {
    addbot();
  }
}

function addIbot() {
  bots.push(new bot());
  bots[bots.length-1].init();
  bots[bots.length-1].brain.loadbrain1(generation[Math.floor(Math.random() * generation.length)].brain);
  bots[bots.length-1].brain.loadbrain2(generation[Math.floor(Math.random() * generation.length)].brain);
  bots[bots.length-1].brain.loadbrain3();
}

function add10Ibot() {
  for (i00=0;i00<10;i00++) {
    addIbot();
  }
}

function worldrun1() {
  for (b in bots) {
    bots[b].hunger -= 10;
    for (f in foods) {
      if (bots[b].x == foods[f].x && bots[b].y == foods[f].y) {
        foods.splice(f, 1);
        bots[b].hunger += 50;
        bots[b].brain.reward = 3;
        bots[b].brain.totalreward += 3;
      }}
    if (bots[b].hunger <= 0) {
      generation.push(bots[b]);
      bots.splice(b,1);
      }
    }
}

function worldrun2() {
  for (x1=0;x1<grid[0];x1++) {
    for (y1=0;y1<grid[1];y1++) {
      //if nothing
      if (String(get(x1*50,y1*50)) == String([179, 179, 179, 255])) {
        worldmap.splice(y1*grid[0]+x1,1,0);
      }
      //if food
      else if (String(get(x1*50,y1*50)) == String([255, 0, 0, 255])) {
        worldmap.splice(y1*grid[0]+x1,1,3);
      }
      //if bot
      else if (String(get(x1*50,y1*50)) == String([0, 0, 0, 255])) {
        worldmap.splice(y1*grid[0]+x1,1,3);  
      }
    }
  }
  delete x1;
  delete y1;
}


//Nerual Network
function neuron() {
  this.init = function(numofconnection,input) {
    this.w = Array(numofconnection).fill(Math.random()*2-1);
    this.b = Math.random()*2-1;
    this.activation = 'sigmoid';
    this.input = input;
    this.output;
  }
  this.activate = function(activation) {
    if (activation == 'sigmoid') {
      this.output = sigmoid(this.output);
    }
  }
  this.fp = function() {
    this.output = 0
    for (e=0;e<this.w.length;e++) {
        this.output += this.w[e] * this.input[e];
    }
    this.output += this.b;
    this.activate(this.activation);
  }
}

function layer() {
  this.init = function(inputlayer,numofneuron) {
    this.inputlayer = inputlayer;
    this.numofneuron = numofneuron;
    this.neuron = [];
    this.outputlayer;
    for (i=0;i<this.numofneuron;i++) {
      this.neuron.push(new neuron())
      this.neuron[this.neuron.length-1].init(inputlayer.length,inputlayer);
    }
  }
  this.fp = function() {
    this.outputlayer = [];
    for (i=0;i<this.numofneuron;i++) {
      this.neuron[i].fp();
      this.outputlayer.push(this.neuron[i].output);
    }
  }
}

function network() {
  this.init = function(dimension) {
    this.dimension = dimension;
    this.layer = [];
    this.inputdata = Array(102).fill(0);
    this.output;
    this.reward = 0;
    this.totalreward = 0;
    this.layer[0] = this.inputdata;
    for (i2=0;i2<(this.dimension.length);i2+=1) {
      this.layer.push(new layer())
      this.layer[this.layer.length-1].init(this.layer[this.layer.length-2],this.dimension[i2]);
    }
  }
  this.loaddata = function(data) {
    this.inputdata = data;
  }

  this.loadbrain1 = function(brain) {
    for (eachlayer=0;eachlayer<brain.dimension.length;eachlayer++) {
      for (eachneuron=0;eachneuron<brain.layer[eachlayer].numofneuron;eachneuron++) {
        this.layer[eachlayer].neuron[eachneuron].w = brain.layer[eachlayer].neuron[eachneuron].w;
        this.layer[eachlayer].neuron[eachneuron].b = brain.layer[eachlayer].neuron[eachneuron].b;
      }
    }
  }

  this.loadbrain2 = function(brain) {
    for (eachlayer=0;eachlayer<Math.floor(brain.dimension.length*(1-mutationrate));eachlayer++) {
      randomlayerid = Math.floor(Math.random() * brain.dimension.length);
      for (eachneuron=0;eachneuron<Math.floor(brain.layer[randomlayerid].numofneuron*(1-mutationrate));eachneuron++) {
        randomneuronid = Math.floor(Math.random() * brain.layer[randomlayerid].numofneuron);
        for (w1=0;w1<Math.floor(this.layer[randomlayerid].neuron[randomneuronid].w.length/2);w1++) {
            this.layer[randomlayerid].neuron[randomneuronid].w[w1] = random.brain.layer[randomlayerid].neuron[randomneuronid].w[w1];
            this.layer[randomlayerid].neuron[randomneuronid].b = random.brain.layer[randomlayerid].neuron[randomneuronid].b;   
        }
      }
    }
  this.loadbrain3 = function() {
    randomlayerid = Math.floor(Math.random() * brain.dimension.length);
    this.layer[randomlayerid].neuron[Math.floor(Math.random() * brain.layer[randomlayerid].numofneuron)].w
    =
    Math.random() - 1;

    this.layer[randomlayerid].neuron[Math.floor(Math.random() * brain.layer[randomlayerid].numofneuron)].b
    =
    Math.random() - 1;
  }
    eachlayer,eachneuron=0;
  }
  this.fp = function() {
    for (i3=1;i3<this.layer.length;i3++) {
      this.layer[i3].fp();
    }
    this.output = this.layer[this.layer.length-1].outputlayer;
  }

//get gradient prep

  //get da/dw & da/db
  this.get_local_gradient = function (location) {
    this.layer[location[0]].neuron[location[1]].dadw = Array(this.layer[location[0]].neuron[location[1]].w.length).fill(0);
    this.layer[location[0]].neuron[location[1]].dadb = [0];
    this.layer[location[0]].neuron[location[1]].fp();
    for (w2=0;w2<this.layer[location[0]].neuron[location[1]].w.length;w2++) {
      this.layer[location[0]].neuron[location[1]].dadw[w2] = sigmoid(this.layer[location[0]].neuron[location[1]].output)
      *(1-sigmoid(this.layer[location[0]].neuron[location[1]].output))*this.layer[location[0]].neuron[location[1]].input[w2];
    }
    this.layer[location[0]].neuron[location[1]].dadb = sigmoid(this.layer[location[0]].neuron[location[1]].output)
    *(1-sigmoid(this.layer[location[0]].neuron[location[1]].output));
    delete w2;
  }

  this.get_layer_gradient = function(location) {
    this.layer[location[0]].neuron[location[1]].dldw = Array(this.layer[location[0]+1].neuron.length).fill(0);
    for (w3=0;w3<this.layer[location[0]+1].neuron.length;w3++) {
      this.layer[location[0]].neuron[location[1]].dldw[w3] = sigmoid(this.layer[location[0]+1].neuron[w3].output)
      *(1-sigmoid(this.layer[location[0]+1].neuron[location[w3]]))*this.layer[location[0]+1].neuron[location[1]];
    }
  }

  this.get_layer_gradient_map = function() {
    this.gradient_map = Array(this.layer.length-2);
    for (l1=1;l1<this.layer.length-1;l1++) {
      for (n1=0;n1<this.layer[l1].neuron.length;n1++) {
        this.layer[l1].neuron[n1].gradient = Array(this.layer[this.layer.length-1].length);
        this.get_local_gradient([l1,n1]);
        //w
        for (w4=0;w4<this.layer[l1].neuron[n1].w.length;w4++) {
          this.layer[l1].neuron[n1].gradient = this.layer[l1].neuron[n1].dadw
        }

        //b


      }
    }

  }

  this.get_gradient = function(location) {
    
  }
  this.learn = function(rate) {
    // for (l=this.dimension.length;l>1;l--) {
    //   for (n=0;n<this.layer[l].numofneuron;n++) {
    //     for (w2=0;w2<this.layer[l].neuron[n].w.length;w++) {
    //         this.layer[l].neuron[n].w[w2] += rate * this.layer[l+1].neuron[n];
    //     }
    //     this.layer[l].neuron[n].b += rate * this.reward;
    //   }
    // }
  }
}

//Neural Network Ends

function infoupdate() {
  numofbot = bots.length;
  numoffood = foods.length;
  infoboard.html(
    'FPS:   '+fps+'<br>'+
    'Generation:  '+gencounter+'<br>'+
    'Number of Foods:  '+numoffood+'<br>'+
    'Number of Bots:  '+numofbot
  );
}

function updatefps() {
  fps = 1 + Number(input1.value());
}

function nextGen(){
  //selection
  basepointer = 0;
  selectionpointer = 0;
  while (generation.length > 10) {
    if (generation[basepointer].totalreward > generation[selectionpointer].totalreward) {
      generation.splice(selectionpointer,1);
    }
    else {
      generation.splice(basepointer,1);
    }
    selectionpointer+=1;
    if (selectionpointer>generation.length) {
      basepointer += 1
      selectionpointer = 0;
    }
  }
  //selection ends
  gencounter += 1;
}

function nextAutoGen() {
  for (i00=0;i00<1;i00++) {
    addfood();
    nextGen();
    add10Ibot();
    add10Ibot();
    addbot();
  }
}

function indexOfMax(arr) {
  if (arr.length === 0) {
      return -1;
  }

  var max = arr[0];
  var maxIndex = 0;

  for (var i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
          maxIndex = i;
          max = arr[i];
      }
  }

  return maxIndex;
}

function sigmoid(x) {
  return (1 / (1 + Math.E ** -x))
}