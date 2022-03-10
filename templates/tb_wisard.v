 `timescale 10us/100ns

module tb_wisard;

// Automatically generated parameters
//__AUTO_PARAMETERS__


// TB signal and variables
reg clk;
reg rst_n;
reg sop, sink_valid, eop;
reg [ADDRESS_WIDTH-1:0] addr;
reg [INDEX_WIDTH-1:0] index;
wire map_sop, map_eop, map_valid;
wire [ADDRESS_WIDTH-1:0] map_addr;
wire [INDEX_WIDTH-1:0] map_index;
wire source_valid;
wire [CLASS_WIDTH-1:0] class_result;
reg [31:0] sample [0:N_RAMS-1];
integer cnt = 0;
reg[20*8:0] fname;
integer fd_o;

// Clock generation
initial begin
   clk = 0;
   while (1) begin
      clk = ~clk;
      #10;
   end
end

initial begin
   rst_n = 0;
   sop = 0;
   sink_valid = 0;
   eop = 0;
   repeat(5) @(negedge clk);
   rst_n = 1;
   repeat(2) @(negedge clk);

   // Load each file containing one input
   repeat (N_INPUTS) begin
      $sformat(fname,"./data/in%04d.txt",cnt);
      $display("Reading file: %s\n", fname);
      $readmemh(fname,sample);
      sop = 1;
      sink_valid = 1;
      index = 0;
      addr = sample[index][ADDRESS_WIDTH-1:0];
      // Send the address for each RAM
      repeat (N_RAMS-1) begin
         @(negedge clk);
         index = index+1;
         sop = 0;
         addr = sample[index][ADDRESS_WIDTH-1:0];
      end
      // End of operation signal should go together with last sample
      eop = 1;
      @(negedge clk);
      eop = 0;
      index = 0;
      sink_valid = 0;

//      while (source_valid==0) @(negedge clk);
//      repeat(2) @(negedge clk);

      cnt=cnt+1;
   end

   repeat(100) @(negedge clk);
   $finish;

end

initial begin
   fd_o = $fopen("./data/y_pred_hw.txt","w"); // HW predictions output file
   repeat(10) @(negedge clk);
   repeat (N_INPUTS) begin
      // Wait for the output
      while (source_valid==0) @(negedge clk);
      repeat(1) @(negedge clk);
      $fwrite(fd_o,"%01d\n",class_result);
   end
   $fclose(fd_o); 
end


//// Mapping instantiation
//wisard_mapping #(.ADDRESS_WIDTH(ADDRESS_WIDTH), .INDEX_WIDTH(INDEX_WIDTH)) 
//wisard_mapping_u0
//(. clk(clk),
//.rst_n(rst_n),
// .sink_sop(sop),
// .sink_valid(sink_valid),
// .sink_eop(eop),
// .addr(addr),
// .index(index),
// .source_sop(map_sop),
// .source_valid(map_valid),
// .source_eop(map_eop),
// .source_addr(map_addr),
// .source_index(map_index));
//
//// Wisard instantiation
//wisard #(.ADDRESS_WIDTH(ADDRESS_WIDTH), .INDEX_WIDTH(INDEX_WIDTH), .N_CLASSES(N_CLASSES), .CLASS_WIDTH(CLASS_WIDTH)) 
//wisard_u0
//(. clk(clk),
//.rst_n(rst_n),
// .sop(map_sop),
// .sink_valid(map_valid),
// .eop(map_eop),
// .addr(map_addr),
// .index(map_index),
// .source_valid(source_valid),
// .class_result(class_result));


// Wisard instantiation
wisard #(.ADDRESS_WIDTH(ADDRESS_WIDTH), .INDEX_WIDTH(INDEX_WIDTH), .N_CLASSES(N_CLASSES), .CLASS_WIDTH(CLASS_WIDTH)) 
wisard_u0
(. clk(clk),
.rst_n(rst_n),
 .sop(sop),
 .sink_valid(sink_valid),
 .eop(eop),
 .addr(addr),
 .index(index),
 .source_valid(source_valid),
 .class_result(class_result));

endmodule
