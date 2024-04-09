<!DOCTYPE HTML>
<?php
require 'util.php';
//session_start();
if(!isset($_SESSION['username'])){
    header('Location: '.'./user.php'); 
}
$test = $_SESSION['username'];

$sql = "select * from voterinfo where vname='".$test."'";
//echo $sql; 
$res = $db->query($sql);
 while($row = $res->fetch_assoc()) {
	 $vname =$row["vname"];
	 $status = $row["status"];
	 $baddress = $row["baddress"];
	 $email = $row["email"];
 }

?>
<html>
<head>
<title>Election a Society and People Category </title>
<!-- for-mobile-apps -->
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="keywords" content="Election Responsive web template, Bootstrap Web Templates, Flat Web Templates, Android Compatible web template, 
Smartphone Compatible web template, free webdesigns for Nokia, Samsung, LG, SonyEricsson, Motorola web design" />
<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
<script type="application/x-javascript"> addEventListener("load", function() { setTimeout(hideURLbar, 0); }, false);
function hideURLbar(){ window.scrollTo(0,1); } </script>
<!-- //for-mobile-apps -->
<link href="assets/css/bootstrap.css" rel='stylesheet' type='text/css' />
<!--<link href='//fonts.googleapis.com/css?family=Raleway:400,100,200,300,500,600,700,800,900' rel='stylesheet' type='text/css'>-->
<!--<link href='//fonts.googleapis.com/css?family=Open+Sans:400,300,300italic,400italic,600,600italic,700,700italic,800,800italic' rel='stylesheet' type='text/css'>-->
<link rel="stylesheet" type="text/css" href="assets/css/style.css">
<!---strat-slider---->
<script src="mfs100-9.0.2.6.js"></script>
<!--script type="text/javascript" src="assets/js/adapter.min.js"></script>
<script type="text/javascript" src="assets/js/vue.min.js"></script>
<script type="text/javascript" src="assets/js/instascan.min.js"></script-->

<script src="https://cdnjs.cloudflare.com/ajax/libs/web3/1.8.1/web3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/truffle-contract@4.0.16/dist/truffle-contract.js"></script>

<!--script src="js/app.js"></script-->

<script src="https://code.jquery.com/jquery-2.2.4.min.js" integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
<script type="text/javascript" src="assets/js/jquery.blockUI.js"></script>
<script type="text/javascript" src="config.js"></script>

<!---//-slider---->
<style>
    .check
{
    opacity:0.5;
	color:#996;
	
}
</style>

<script >
    console.log(gOptions.contractaddress);
    function processResult(str){
    //alert(str);
    $xmlDoc = $.parseXML( str );
  
	//console.log(gOptions.contractaddress);
  uid = $xmlDoc.children[0].getAttribute('uid');
  $("#aid").val(uid);
    }
	
var ename;
var etype;
var esub;
var aadharno;
var file;
var candidateid="";
        var flag =0;
        var quality = 60; //(1 to 100) (recommanded minimum 55)
        var timeout = 10; // seconds (minimum=10(recommanded), maximum=60, unlimited=0 )

</script>
<script>
  $( function() {
      //aadhar serach
  $("#aadSearch").click(function(){
		aid = $("#aid").val();
		if(!aid){
		alert('Please enter your Aadhar Number');
		return;
	}
	//get voter details using aadhar
	
	$.getJSON("getVDetail.php?aid="+aid, function(result){

                err = result['error'];
                if(err){
                    alert(err);
                    return;
                }
                elec=result['election'];
				console.log("election:"+elec);
                for(i=0;i<elec.length;i++){
                  elrow=elec[i];
                    console.log(elec[i]);
                   $("#eldiv").css('display','block');     
               				
                $("#elname").html(elrow['ename']);
                ename=elrow['ename'];
                $("#eldate").html(elrow['edate']);
                $("#elcons").html(elrow['subtype']);
				//-------------------------------------
                etype=elrow['etype'];    //used in cast vote 
                esub = elrow['subtype'];  //used in cast vote
                $("#area").val(elrow['subtype']);            
			}
        console.log(i);
        if(ename!=0){
            //$("#confirmdiv").css("display","block");
			document.getElementById("btnCapture").disabled=false;
			//getcandidates(ename);
        }

        });

});

});
 
</script>

</head>
<body>
<!-- header -->
	<div class="header_bg">
		<div class="container">
			<!-----start-header----->
			<div class="header">
				
				<nav class="navbar navbar-default">
					<!-- Brand and toggle get grouped for better mobile display -->
					<div class="navbar-header">
					  <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1">
						<span class="sr-only">Toggle navigation</span>
						<span class="icon-bar"></span>
						<span class="icon-bar"></span>
						<span class="icon-bar"></span>
					  </button>
					</div>

					<!-- Collect the nav links, forms, and other content for toggling -->
					<div class="collapse navbar-collapse nav-wil" id="bs-example-navbar-collapse-1">
						<ul class="nav navbar-nav">
							
                                                        <li><a href="index.php">Home</a></li>                                                        
                                                        <li class="act"><a href="voting.php">Voting</a></li>														
                                                        <li><a href="results.php">Results</a></li>
							                            <li><a href="logout.php">Logout</a></li>

						</ul>
					</div><!-- /.navbar-collapse -->
					
				</nav>
			</div>
		</div>
	</div>
	<div class="header_bottom">
	</div>

	<div class="typo">
	
		<div class="container">
			<h3 class="title">Welcome for Online Voting</h3>
			<p class="nihil">Vote For Real Government.</p>
			<div class="row">
			<div class="col-md-9">
			<h3 class="hdg">Please enter your Aadhar Number</h3>
			<h4 id="h4.-bootstrap-heading"><b>Adhar Number</b></h4><input type="text" style="border: none; border-bottom: 2px solid steelblue;width: 60%;height:50px" id="aid"/><span style="font-size: 12px" class="glyphicon glyphicon-search" id="aadSearch"></span>
			<br></br>
			<br></br>
			<h4 id="h4.-bootstrap-heading"><b>Constituency</h4>
			<input type="text" readonly style="border: none; border-bottom: 2px solid steelblue;width: 60%;height:50px"id="area"/></td>
			<br></br>
			<br></br>
			<h4 id="h4.-bootstrap-heading"><b>Election</h4>
            
			<div style="float:left" id="elname"></div>
            <br></br>
			
			<h4 id="h4.-bootstrap-heading"><b>Date :</h4>    
            
            <div><span id="eldate"></span></div>
			
			
            
                
                <!--div style="width:40%">
            Constituency :<span  id="elcons"></span>
            </div-->
            
			</div>
			
			<div class="col-md-3">
			<h3>Cameras</h3>
			<div id="container">
				<video width="250" height="250" autoplay="true" id="video">
				</video>
				</div>
				<!--button onclick="capture()">Capture</button-->  
				 <input type="button" id="btnCapture" value="Capture" disabled="true" class=" capturebuttonpadding btn btn-primary btn-lg submit_buttom_padding" style="margin-top:20px"/>		
				 <br> </br>
				<canvas id="canvas" style="overflow:auto"></canvas>
				<input id="btnConfirm" value="Confirm" type="hidden" class=" capturebuttonpadding btn btn-primary btn-lg submit_buttom_padding" style="margin-top:20px"/>		
				<br></br>	
				<a id="link"></a>
			</div>
			</div>
     
                        <form action="" method="post">
                        
                        <br> 
			<div class="row" style="height: 170px;display: none" id="chpersondiv">
				<h3>Choose a Person</h3>
                          <!--Dynamically generate candidate list using js-->
				 </div>
         
                            <br>
                            <p class="one"><div style="background-color:steelblue;color:white ;height: 50px;width: 100%;border: 2px solid steelblue;"></div>
                            <!--input type="submit" value="CLICK HERE TO POLL YOUR VOTE" style="background-color:steelblue;color:white ;height: 50px;width: 100%;border: 2px solid steelblue;"--></p>
		</form>	
			</div>	
						<!-- /.row -->
		
		<div id="wrapperProposals">
      </div>
			
		
		</div>
	

<!-- //Vote Casting -->
<script type="text/javascript"> 
  
    $(document).ready(function() { 
 
        $('#yes').click(function() { 
            console.log("Aadhar "+aadharno);
            console.log("ename " +ename);
            console.log("candidate "+candidateid);
            // update the block message 
            $.blockUI({ message: "<h1>Registering your Vote</h1>" }); 
			
		$.getJSON('./contract/Votingsol.json', function (data) {	
		console.log("contract");		
		console.log(data.abi);
		const web3 = new Web3();
		web3.setProvider(new Web3.providers.WebsocketProvider('ws://localhost:8545'));      
		web3.eth.net.isListening()
		.then(() => {
		console.log('is connected');
		PNBContract = new web3.eth.Contract(data.abi,gOptions.contractaddress);		
		
		var baddress = '<?php echo $baddress; ?>'
		PNBContract.methods.vote(candidateid,ename).send( {from:baddress},function(err, res){
  			console.log("inside voting");
               if (err) {
                   console.log("error in set!")                   
               }
			   else
			   {
				console.log(res);
				AddVote(res);
			   }
			});           		
		})
		.catch(e => {
		console.log('Wow. Something went wrong: '+ e);
		alert("Ganache Server not Running");
		});				
		}); 
    }); 
	
	function AddVote(res) {
	var txhash=res;
	url1="castVote.php";
          data1={'aadhar':aadharno,'ename':ename,'candidate':candidateid,'etype':etype,'esubtype':esub,'txhash':txhash};
            $.post(url1,data1,function(){},'json')
                .done(function (response){				
				console.log(response.status);
				//sendMail(txhash);  //enable to send mail
				alert ("Successfully Voted");
                location.href = './index.php';
            }).fail(function (jqxhr, textStatus, error){
              var err = textStatus + ", " + error;
        console.log("Request Failed: " + err);
        });
        }
 
	$('#no').click(function() { 
            $.unblockUI(); 
            return false; 
        });	

//function to capture the face.	
	$("#btnCapture").click(function() {
	var canvas = document.getElementById('canvas');     
    var video = document.getElementById('video');
    canvas.width = 250;
    canvas.height = 250;
    canvas.getContext('2d').drawImage(video, 0, 0, 250,250);  
	document.getElementById("btnConfirm").type="button";	
	file = dataURLtoFile(canvas.toDataURL('image/jpg'), "verify.png");
	//var res = getCandidates();
 });
 
 function dataURLtoFile(dataurl, filename) {

		var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1], bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(
				n);

		while (n--) {
			u8arr[n] = bstr.charCodeAt(n);
		}

		return new File([ u8arr ], filename, {
			type : mime
		});
	}

//function to confirm
$("#btnConfirm").click(function(){
 
    aid = $("#aid").val();
    aadharno=aid;
	if (ename != "")
	{
		var form = new FormData();
		form.append("name", file.name);		
		form.append("file", file);

		var settings = {
			"url" : "http://localhost:5001/",
			"method" : "POST",
			"timeout" : 0,
			"processData" : false,
			"mimeType" : "multipart/form-data",
			"contentType" : false,
			"data" : form			
		};
		
		$.ajax(settings).done(function(response) {
			console.log("---py-res---"+response);
			 console.log(response.status);
			 var obj=JSON.parse(response);
			console.log(isEmpty(obj));
			if(!isEmpty(obj))
			{				
			if (obj[0].status)
			{			
			getCandidates();			
			}
			else
			{
				alert("Voter face not matched");
			}
			}
			else
			{
				alert("No Proper Face Detected");
			}
		}).fail(function (jqXHR, textStatus) {
			console.log(jqXHR);
			});;
	}		
   			
});	

function isEmpty(obj) {
    return Object.keys(obj).length === 0;
}

 function getCandidates()
{
	$.getJSON('./contract/Votingsol.json', function (data) {			
		console.log("contract");
		console.log(data.abi);
		
		var web3 = new Web3(new Web3.providers.HttpProvider('http://localhost:8545'));
				
		web3.eth.net.isListening()
		.then(() => {
			console.log('is connected');
			PNBContract = new web3.eth.Contract(data.abi,gOptions.contractaddress);
			PNBContract.methods.getCandidates(ename).call(function(err, res){
		      		if (res) {
						console.log("insidename");
						console.log(res.length);
						var out =res;
						console.log(res);
						var iterator = res.values();
					$("#chpersondiv").css("display","block");
					$("#chpersondiv").css("position","relative");
					//console.log(iterator);
					for (let elements of iterator) {
						if (elements != "")
						{					
						var text = elements;						
						const myArray = text.split("-");
						var candidate = myArray[0];
						var symbol = myArray[1];
						var candidateid = myArray[2];
						
				document.getElementById('chpersondiv').style.height='150px';
				var inputTag = document.createElement("input");  
				inputTag.setAttribute("id", candidateid);        
				inputTag.setAttribute("type","button");
				
				inputTag.setAttribute("style","height:100px;width:150px; background-size:cover; background-image:url('../An_OnlineVotingQR_Admin/upImages/"+symbol+"');");
				inputTag.onclick=function(){your_function_name(this.id)};
				var div = document.createElement("div");
				inputTag.setAttribute("value", candidate);
				inputTag.innerHTML=candidate;
				document.getElementById('chpersondiv').appendChild(inputTag);
				var label = document.createElement("label");
				label.style.marginRight = "15px";
				document.getElementById('chpersondiv').appendChild(label);
						}
					}					
		       	}
				if(err)
				{
					console.log(err);
				}
      		 	});
			})
		.catch(e => {
		console.log('Wow. Something went wrong: '+ e);
		alert("Ganache Server not Running");
		});
		
		});
}

function your_function_name(clicked){
  						$(".img-check").removeClass("check");
						$.blockUI({ message: $('#question'), css: { width: '275px' } });
                        candidateid=clicked;
                        $(this).parent().children("img").toggleClass("check");
                        console.log('candidate id'+candidateid);								
		  }
		  
//send mail after successfull voting
function sendMail(res)	{ 
	var data = {
        'vname': '<?php echo $vname; ?>',
        'email': '<?php echo $email; ?>',
		'hash' : res
    };
	$.ajax({ 
        url: 'mailto.php', 
        data: data,
        type: 'POST',
        success: function (data) {
			res = $.parseJSON(data);
			console.log("message"+res);
			console.log(res['message']);
			/*if(res['status']){
				console.log(data.error);
			}
			else
			{
				console.log(data.message);
			}*/
	}
   
	}) 
	}

	});
  
</script> 
<script>
 var video = document.querySelector("#video");
 if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
          video.srcObject = stream;
        })
        .catch(function (err0r) {
          console.log("No media-Something went wrong!");
        });
 }

</script>

<div id="question" style="display:none; cursor: default"> 
        <h1>Confirm to Cast your Vote.</h1> 
        <input type="button" id="yes" value="Confirm" /> 
        <input type="button" id="no" value="No" /> 
</div> 
<!-- footer -->
	<div class="footer">
		<div class="container">
			
			
			<div class="footer-copy">
				<p>All rights reserved | www.onlineelectionvoting.com</a></p>
			</div>
		</div>
	</div>
<!-- //footer -->
		<!-- scroll_top_btn -->
		<script type="text/javascript" src="assets/js/move-top.js"></script>
		<script type="text/javascript" src="assets/js/easing.js"></script>
	    <script type="text/javascript">
			$(document).ready(function() {
			
				var defaults = {
		  			containerID: 'toTop', // fading element id
					containerHoverID: 'toTopHover', // fading element hover id
					scrollSpeed: 1200,
					easingType: 'linear' 
		 		};
				
				
				$().UItoTop({ easingType: 'easeOutQuart' });
				
			});
		</script>
		 <a href="#" id="toTop" style="display: block;"><span id="toTopHover" style="opacity: 1;"></span></a>
		<!--end scroll_top_btn -->
<!-- for bootstrap working -->
	<!--script type="text/javascript" src="assets/js/camera/app.js"></script-->
	<script type="text/javascript" src="assets/js/bootstrap-3.1.1.min.js"></script>
	

	
<!-- //for bootstrap working -->
</body>
</html>